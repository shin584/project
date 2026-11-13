"""Reader checks against the real prepared data; skipped where it is absent.

These are the one check the synthetic fixtures cannot give: that the reader's
profile definitions, head selection and row join reproduce the figures the
analyses themselves recorded. The bulk data is outside Git, so everywhere else
these skip and the fixture tests carry the defence.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from xai_test.nt_xai_demo import reader as rd


def _reader(cls, *needed: str):
    data = rd.DEFAULT_DATA_DIR
    if not (data / "manifest.json").exists():
        pytest.skip("실제 데모 자료 없음: `prepare`를 먼저 실행")
    for name in needed:
        if not (data / name).exists():
            pytest.skip(f"git 밖 bulk 자료 없음: {name}")
    return cls(data)


def test_spcas9_reader_reproduces_the_recorded_generalisation_token_profile():
    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ism/ism.npz")
    head = spcas9.primary_head
    recorded = spcas9.validation_figures("generalisation", head)

    rows = sorted(spcas9.ig_subset_rows("generalisation"))
    profiles = spcas9.ism_profiles(rows, head)
    tokens = rd.aggregate_to_tokens(profiles, spcas9.token_boundaries).mean(axis=0)

    assert len(rows) == recorded["rows"]
    assert tokens == pytest.approx(recorded["ism_token_profile"], rel=1e-6)
    # The PAM-bearing token is the most sensitive one, at this layer and head.
    assert int(np.argmax(tokens)) == len(spcas9.token_names) - 1


def test_spcas9_reader_reproduces_the_recorded_pam_library_token_profile():
    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ism/ism.npz")
    head = spcas9.primary_head
    recorded = spcas9.validation_figures("pam_library", head)

    rows = sorted(spcas9.ig_subset_rows("pam_library"))
    profiles = spcas9.ism_profiles(rows, head)
    tokens = rd.aggregate_to_tokens(profiles, spcas9.token_boundaries).mean(axis=0)

    assert tokens == pytest.approx(recorded["ism_token_profile"], rel=1e-6)
    assert not np.allclose(
        recorded["ism_token_profile"],
        spcas9.validation_figures("generalisation", head)["ism_token_profile"],
    )


def test_spcas9_ism_and_ig_populations_are_all_test_rows():
    """The measured premise behind showing the split role as a cheap safeguard."""
    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ism/ism.npz")

    for axis in (spcas9.ism_rows, spcas9.ig_rows):
        roles = {spcas9.split_role(int(row_id)) for row_id in axis.ids}
        assert roles == {"test"}, (axis.source, roles)


def test_spcas9_wt_prediction_matches_the_predictions_table_row_by_row():
    """ISM's own WT prediction is the check that catches a mis-join."""
    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ism/ism.npz")
    head = spcas9.primary_head

    sample = [int(row_id) for row_id in spcas9.ism_rows.ids[::400]]
    for row_id in sample:
        assert spcas9.wt_prediction(row_id, head) == pytest.approx(
            spcas9.prediction(row_id, head), rel=1e-5, abs=1e-6
        )


def test_cas12a_reader_reproduces_the_recorded_position_profile():
    cas12a = _reader(rd.Cas12aReader, "cas12a/ism.npz")
    recorded = cas12a.position_profile("all")

    rows = [int(row_id) for row_id in cas12a.ism_rows.ids]
    profiles = np.stack([cas12a.ism_profile(row_id) for row_id in rows])

    assert len(rows) == recorded["n_available"]
    assert profiles.mean(axis=0) == pytest.approx(
        recorded["mean_abs_raw_delta"], rel=1e-9
    )
    # Position 7, the last base of the 5' PAM (4-7), is the most sensitive.
    assert int(np.argmax(profiles.mean(axis=0))) == 7


def test_cas12a_labels_and_predictions_are_fractions_and_the_claim_limit_is_present():
    cas12a = _reader(rd.Cas12aReader)

    row = int(cas12a.prediction_rows.ids[0])
    record = cas12a.prediction(row)

    assert 0.0 <= record["label"] <= 1.5
    assert cas12a.metrics_for("all")["count"] == 1500
    assert (
        cas12a.metrics_for("TTTV")["count"] + cas12a.metrics_for("TTTT")["count"]
        == 1500
    )
    assert cas12a.rollout_degeneracy("all") == {
        "near_identical_query_count": 1500,
        "n_accepted": 1500,
    }
    assert "[범위와 한계]" in cas12a.claim_limit
    assert "분모" in cas12a.claim_limit


def test_spcas9_claim_limit_is_one_statement_across_its_artifacts():
    spcas9 = _reader(rd.SpCas9Reader)

    assert "physical DNA/protein" in spcas9.claim_limit
    assert spcas9.analysed_heads == ("SpCas9", "SpCas9-NG", "evoCas9")
    assert spcas9.ism_heads.width == 9 and spcas9.ig_heads.width == 3


def test_the_real_spcas9_report_shows_only_the_three_analysed_heads():
    """The six stored-but-unanalysed heads must not reach the screen (#155).

    The synthetic fixture pins this too, but the real ISM archive is where the
    nine actually live, and the head names come from it.
    """
    from xai_test.nt_xai_demo import spcas9_screen

    spcas9 = _reader(rd.SpCas9Reader)
    report = spcas9_screen.report(spcas9)

    unanalysed = set(spcas9.ism_heads.names) - set(spcas9.analysed_heads)
    assert len(unanalysed) == 6
    for head in unanalysed:
        assert head not in report, head
    for head in spcas9.analysed_heads:
        assert head in report, head


def test_the_real_spcas9_report_quotes_the_per_head_floor_not_the_pooled_one():
    from xai_test.nt_xai_demo import spcas9_screen

    spcas9 = _reader(rd.SpCas9Reader)
    report = spcas9_screen.report(spcas9)
    pooled = float(spcas9.noise_floor["pooled"]["median"])

    for head in spcas9.analysed_heads:
        own = float(spcas9.noise_floor_by_head(head)["median"])
        assert f"{own:.3g}" in report, head
    # The pooled median is a fourth number none of the three heads is, and
    # printing it would invite reading one floor for all of them.
    assert f"{pooled:.3g}" not in report and f"{pooled:.4g}" not in report


def test_the_real_case_tables_keep_their_own_identifiers():
    """18 SpCas9 cases by `case_id`, 15 Cas12a cases by `source_row` + `role`."""
    spcas9 = _reader(rd.SpCas9Reader)
    cas12a = _reader(rd.Cas12aReader)

    assert len(spcas9.case_ids()) == 18
    assert spcas9.case("GEN-HEADGAP-02")["row_index"] == 17068
    assert len(cas12a.case_ids()) == 15
    assert all(":" in identifier for identifier in cas12a.case_ids())


def test_gen_headgap_02_is_the_only_real_case_carrying_ig():
    """The demonstration path, measured: 1 of 18 falls inside the 400-row list.

    The other 17 are the "이 행은 IG 고정 명단 400행에 없음" path, and the
    annotation's own `ig_coverage` has to agree with what the IG row axis says.
    """
    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ig/ig.npz")

    with_ig = [
        case_id
        for case_id in spcas9.case_ids()
        if spcas9.case_annotation(case_id).get("ig") is not None
    ]

    assert with_ig == ["GEN-HEADGAP-02"]
    assert spcas9.has_ig(17068)
    coverage = spcas9.cases_annotated["ig_coverage"]
    assert coverage["cases"] == 18 and coverage["cases_with_ig"] == 1


def test_every_real_spcas9_case_row_is_a_test_row():
    """The split-role safeguard over the rows a speaker will actually open."""
    spcas9 = _reader(rd.SpCas9Reader)

    roles = {
        spcas9.split_role(int(spcas9.case(case_id)["row_index"]))
        for case_id in spcas9.case_ids()
    }

    assert roles == {"test"}


def test_cas12a_reader_reads_the_recorded_token_spans_and_pam():
    """Five 6-mers then four single bases over 34bp, PAM at [4, 8] - as recorded."""
    cas12a = _reader(rd.Cas12aReader, "cas12a/layout.json")

    spans = cas12a.token_spans
    assert len(spans) == 9
    assert [end - start for start, end in spans] == [6] * 5 + [1] * 4
    assert spans[0][0] == 0 and spans[-1][1] == 34
    assert cas12a.pam_span == (4, 8)


def test_the_real_case_screens_open_for_all_fifteen_numbers_in_one_screen():
    """Every number opens, rows join (sequence = ISM reference), ~25 lines each."""
    from xai_test.nt_xai_demo import cas12a_screen

    cas12a = _reader(
        rd.Cas12aReader, "cas12a/ism.npz", "cas12a/ig.npz", "cas12a/rollout.npz"
    )
    assert cas12a_screen.number_range(cas12a) == (
        "A1–A5 low_error · B1–B5 high_error · C1–C5 out_of_range"
    )
    for number, case in cas12a_screen.numbered_cases(cas12a).items():
        text = cas12a_screen.case_screen(cas12a, number)
        assert text.startswith(
            f"{number}  {case['role']}  source_row {case['source_row']}"
        )
        assert len(text.splitlines()) <= 25
        # The IG values on screen are the analysis' own recorded DNA masses.
        recorded = cas12a.case_annotation(f"{case['source_row']}:{case['role']}")
        np.testing.assert_allclose(
            cas12a.ig_dna_mass(int(case["source_row"])),
            recorded["methods"]["ig"]["summary"]["dna_signed_mass"],
            rtol=1e-6,
        )
    assert len(cas12a_screen.report(cas12a).splitlines()) < 20


def test_the_real_sequences_open_the_number_screens_body_over_all_1500_rows():
    """Every case's 34bp window opens its number's body; the index holds 1,500."""
    from xai_test.nt_xai_demo import cas12a_screen

    cas12a = _reader(
        rd.Cas12aReader, "cas12a/ism.npz", "cas12a/ig.npz", "cas12a/rollout.npz"
    )
    index = cas12a.sequence_index
    assert index.stored_rows == 1500

    def body(text):
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("실측 "))
        return lines[start:-1]

    for number, case in cas12a_screen.numbered_cases(cas12a).items():
        sequence = cas12a.sequence(int(case["source_row"]))
        by_sequence = cas12a_screen.sequence_screen(cas12a, sequence)
        assert by_sequence.startswith(f"서열 입력  source_row {case['source_row']}")
        assert number in by_sequence.splitlines()[0]
        assert body(by_sequence) == body(cas12a_screen.case_screen(cas12a, number))
        assert len(by_sequence.splitlines()) <= 26


def test_the_real_spcas9_screens_open_for_every_generalisation_case():
    """All nine numbers open, and exactly one of them carries IG (#173).

    The counts here are the artifacts' own facts, not the demo's: nine
    generalisation cases, one of which (`GEN-HEADGAP-02`) happens to sit in
    the frozen 400-row IG subset. If a re-run of the analyses changed either,
    the screen's standing IG sentence would be wrong and this fails.
    """
    from xai_test.nt_xai_demo import spcas9_screen

    spcas9 = _reader(
        rd.SpCas9Reader,
        "spcas9/v3_ism/ism.npz",
        "spcas9/v3_ig/ig.npz",
        "spcas9/v3_rollout/rollout.npz",
    )
    numbered = spcas9_screen.numbered_cases(spcas9)
    assert set(numbered) == {
        f"{letter}{rank}" for letter in "ABC" for rank in (1, 2, 3)
    }

    with_ig = []
    for number, case in numbered.items():
        text = spcas9_screen.case_screen(spcas9, number)
        row = int(case["row_index"])
        assert text.startswith(f"{number}  {case['case_id']}")
        assert f"row_index {row}" in text
        # The 30bp window and its bar line up column for column.
        sequence = next(
            line for line in text.splitlines() if line.startswith("서열  ")
        ).removeprefix("서열  ")
        assert sequence == spcas9.reference_sequence(row) and len(sequence) == 30
        assert len(text.splitlines()) <= 26
        if spcas9_screen.IG_SAMPLE_NOTE not in text:
            with_ig.append(number)

    assert len(with_ig) == 1
    assert numbered[with_ig[0]]["case_id"] == "GEN-HEADGAP-02"


def test_the_checklists_demonstration_number_is_the_case_that_carries_ig():
    """`spcas9> C2` and `GEN-HEADGAP-02` have to be the same row (#175).

    The checklist tells the meeting to type a number and names the case beside
    it. Nothing in the demo keeps those two in step - the number comes from the
    selection's rank order - so if a re-run renumbered the cases, the meeting
    would open a case with no IG while reading a line that promises it.
    `test_docs.py` cannot see this: it has no data to resolve a number with.
    """
    import re
    from pathlib import Path

    from xai_test.nt_xai_demo import spcas9_screen

    spcas9 = _reader(rd.SpCas9Reader)
    checklist = (Path(rd.__file__).parent / "demo_checklist.md").read_text(
        encoding="utf-8"
    )

    numbers = {
        number
        for number in re.findall(r"`spcas9> ([A-C]\d)`", checklist)
        if "GEN-HEADGAP-02" in checklist.split(f"`spcas9> {number}`", 1)[1][:200]
    }
    assert len(numbers) == 1, f"IG 사례를 가리키는 번호가 하나가 아니다: {numbers}"

    numbered = spcas9_screen.numbered_cases(spcas9)
    assert numbered[numbers.pop()]["case_id"] == "GEN-HEADGAP-02"


def test_the_real_spcas9_headgap_screens_show_two_heads():
    from xai_test.nt_xai_demo import spcas9_screen

    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ism/ism.npz")
    for number, case in spcas9_screen.numbered_cases(spcas9).items():
        heads = spcas9_screen.heads_shown(spcas9, case)
        expected = 2 if case["criterion"] == "HEADGAP" else 1
        assert len(heads) == expected, number
        assert heads[0] == spcas9.primary_head


def test_the_real_spcas9_report_states_both_populations_with_denominators():
    """The selection pool and the agreement subset are different populations."""
    from xai_test.nt_xai_demo import spcas9_screen

    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ism/ism.npz")
    text = spcas9_screen.report(spcas9)

    assert "선정 모집단 = generalisation 895행 / 818 타깃" in text
    assert "분모 200행 / 193 타깃" in text  # the agreement subset's own
    assert "다른 모집단이다" in text
    assert len(text.splitlines()) < 25


def test_the_real_spcas9_lookup_is_the_generalisation_stratum_cases_json_counts():
    """#174: the 30bp lookup is generalisation ∩ ISM, and that is cases.json's pool.

    Generalisation is `v3_strata`'s cut - non-library **and** NGG, on the rows
    where all three analysed heads are measured. On this data that is exactly
    the 895 rows / 818 targets `cases.json` records, so the lookup and the
    case screen stand on one population. Non-library without the NGG cut
    would be 2,564 rows, 1,669 of them in neither stratum.
    """
    from xai_test.nt_xai_demo import spcas9_screen

    spcas9 = _reader(
        rd.SpCas9Reader,
        "spcas9/v3_ism/ism.npz",
        "spcas9/v3_ig/ig.npz",
        "spcas9/v3_rollout/rollout.npz",
    )
    pool = spcas9.case_strata["generalisation"]
    index = spcas9.sequence_index
    rows = spcas9.generalisation_rows

    assert index.stored_rows == len(rows) == int(pool["rows"]) == 895
    sequences = [spcas9.reference_sequence(row) for row in rows]
    assert len({sequence[4:24] for sequence in sequences}) == 818
    assert int(pool["distinct_protospacers"]) == 818
    assert all(row in spcas9.ism_rows for row in rows)
    assert all(sequence[25:27] == "GG" for sequence in sequences)
    assert all(
        spcas9.label(row, head) is not None
        for row in rows
        for head in spcas9.analysed_heads
    )
    # Replicates: widening 20bp to 30bp does not separate a single one.
    counts = Counter(sequences)
    assert index.distinct == 818
    assert sum(1 for n in counts.values() if n > 1) == 20
    assert max(counts.values()) == 5

    # A replicated window shows its smallest row with the duplicate line.
    sequence, _ = counts.most_common(1)[0]
    shared = sorted(row for row in rows if spcas9.reference_sequence(row) == sequence)
    lines = spcas9_screen.screen_for(spcas9, sequence).splitlines()
    assert lines[0].startswith(f"서열 입력  row_index {shared[0]}  generalisation")
    assert lines[1].startswith(f"이 타깃은 {len(shared)}행 중복(세포주 반복)")


def test_the_real_spcas9_case_sequences_open_a_screen_on_the_smallest_row():
    """A case's own window opens its number's body - unless it is replicated.

    Eight of the nine case rows are the only row holding their window, and
    their window opens exactly the number's screen. `C3`'s row (20544) is one
    of the 20 replicated keys, so its window resolves to the smallest row
    sharing it (9343) and shows that row's values with the duplicate line -
    the rule #172 set, applied here rather than excepted for a case.
    """
    from xai_test.nt_xai_demo import spcas9_screen

    spcas9 = _reader(
        rd.SpCas9Reader,
        "spcas9/v3_ism/ism.npz",
        "spcas9/v3_ig/ig.npz",
        "spcas9/v3_rollout/rollout.npz",
    )

    def body(text):
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("  SpCas9 "))
        return lines[start:-1]

    replicated = []
    for number, case in spcas9_screen.numbered_cases(spcas9).items():
        row = int(case["row_index"])
        sequence = spcas9.reference_sequence(row)
        text = spcas9_screen.screen_for(spcas9, sequence)
        shown = spcas9.sequence_index.find(sequence)
        assert text.splitlines()[0].startswith(
            f"서열 입력  row_index {shown.row}  generalisation"
        )
        assert len(text.splitlines()) <= 27
        if shown.row == row:
            assert number in text.splitlines()[0]
            assert body(text) == body(spcas9_screen.case_screen(spcas9, number))
        else:
            # The shown row is not the case's, so the screen does not claim it.
            replicated.append((number, row, shown.row))
            assert number not in text.splitlines()[0]
            assert text.splitlines()[1].startswith("이 타깃은 ")

    assert replicated == [("C3", 20544, 9343)]
