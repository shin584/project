"""The SpCas9 case screen and short report, against golden output (#173).

The fixture is built so the misreads ADR 0012 names show up in the text:

* **Head by name.** The ISM archive stores nine heads and the IG archive three
  **in a different order**, and here each head carries its own ISM spike, its
  own IG values and its own recorded agreement median. A screen that reused
  one artifact's axis position on the other, or showed the primary head's
  numbers under a second head's name, prints different text.
* **Percent vs fraction.** The split files store labels as percent and
  `cases.json` stores them as fractions. The shown labels are the fractions,
  and `_checked_row` compares the two sources before anything is drawn, so a
  factor of 100 is a refusal rather than a plausible-looking screen.
* **`row_index` join.** The IG population is a subset of the ISM one and their
  orders differ, and each row's rollout block is its own; a join by array
  position prints another row's numbers.
* **WT slot.** The spike is written only into the non-WT slots, so a bar that
  kept the reference slot's true zero in its max would flatten the peak.

The golden strings are the whole screen, so any change to what the meeting
sees is a visible diff here.
"""

from __future__ import annotations

import numpy as np
import pytest

from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo import spcas9_screen as screen
from xai_test.nt_xai_demo.test import reader_fixtures as fx

#: Where each head's ISM peak is put and how big it is, per row. The position
#: differs per head, so the two bars of a HEADGAP screen cannot both be the
#: primary head's; the *magnitude* differs too, so the per-head peak figures
#: cannot both be read off one head. The second head's peak is an order
#: smaller than the primary's, as on the real data (C1: 0.454 against 0.035),
#: the per-head figures exist for - both bars draw a full block regardless.
SPIKES = {
    (44, "SpCas9"): (3, 0.9),
    (5, "SpCas9"): (11, 0.9),
    (11, "SpCas9"): (26, 0.9),
    (11, "evoCas9"): (7, 0.6),
}


def _spike(spcas9_dir, row: int, head: str, position: int, value: float) -> None:
    """Give one row and head an ISM peak that no other row or head has."""
    path = spcas9_dir / "v3_ism/ism.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    at = int(np.flatnonzero(arrays["row_index"] == row)[0])
    head_index = fx.ISM_HEADS.index(head)
    reference = arrays["reference_base"][at, position]
    for base in range(len(fx.BASES)):
        if base != reference:  # the WT slot keeps its true zero
            arrays["delta"][at, position, base, head_index] = -value
    np.savez(path, **arrays)


@pytest.fixture
def reader(tmp_path):
    data = fx.build_data_dir(tmp_path / "data")
    for (row, head), (position, value) in SPIKES.items():
        _spike(data / "spcas9", row, head, position, value)
    return rd.SpCas9Reader(data)


GOLDEN_A1 = """\
A1  GEN-ERRLO-01  ERRLO  row_index 44  generalisation
  SpCas9     실측 0.1   예측 0.1   오차 0

서열  CATGCATGCATGCATGCATGCATGCATGCA
ISM   ▃▃▃█▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃  SpCas9
                              [----]  pam_plus_right_flank 24-29
      막대 = 위치별 대체염기 3개 중 최대 |Δ| (WT 칸 제외) · 각 막대는 자기 최대에 맞춰 그린다 (fraction)
      SpCas9 최대 0.900 @ 위치 3

IG (토큰 5개, 위치 순, signed)
  IG는 층당 200행 표본 — 이 행은 비대상 (고정 명단 2행, 사례 선정 전에 자기 seed로 고정)

Attention Rollout (토큰별 참조되는 정도, 토큰 query 5개 평균, head 축이 없어 head별 값이 아니다 (ADR 0008))
  t1 85.000  t2 86.000  t3 87.000  t4 88.000  t5 89.000
  모델 내부 정보 의존성이며 물리적 결합의 근거가 아님

출처: 층=generalisation · head=SpCas9 · 통계=ISM 최대 |Δ|(fraction), IG signed token attribution, rollout 토큰 열 평균 · 분모=이 행 1행 · 선정=ERRLO rank 1, generalisation 895행 / 818타깃 중
"""

GOLDEN_C2 = """\
C2  GEN-HEADGAP-02  HEADGAP  row_index 11  generalisation  (head 2개: evoCas9가 SpCas9 대비 오차 차이 최대)
  SpCas9     실측 0.4   예측 0.1   오차 0.3
  evoCas9    실측 0.56   예측 0.18   오차 0.38

서열  TGCATGCATGCATGCATGCATGCATGCATG
ISM   ▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃█▃▃▃  SpCas9
ISM   ▆▆▆▆▆▆▆█▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆  evoCas9
                              [----]  pam_plus_right_flank 24-29
      막대 = 위치별 대체염기 3개 중 최대 |Δ| (WT 칸 제외) · 각 막대는 자기 최대에 맞춰 그린다 (fraction)
      SpCas9 최대 0.900 @ 위치 26 · evoCas9 최대 0.600 @ 위치 7

IG (토큰 5개, 위치 순, signed, 표준화 출력 (fraction 아님))
  t1  0-5    left_flank_plus_protospacer_5prime  SpCas9 +11.000  evoCas9 +10.000
  t2  6-11   protospacer_6_11                    SpCas9 +11.100  evoCas9 +10.100
  t3  12-17  protospacer_12_17                   SpCas9 +11.200  evoCas9 +10.200
  t4  18-23  protospacer_18_23                   SpCas9 +11.300  evoCas9 +10.300
  t5  24-29  pam_plus_right_flank                SpCas9 +11.400  evoCas9 +10.400

Attention Rollout (토큰별 참조되는 정도, 토큰 query 5개 평균, head 축이 없어 head별 값이 아니다 (ADR 0008))
  t1 35.000  t2 36.000  t3 37.000  t4 38.000  t5 39.000
  모델 내부 정보 의존성이며 물리적 결합의 근거가 아님

출처: 층=generalisation · head=SpCas9, evoCas9 · 통계=ISM 최대 |Δ|(fraction), IG signed token attribution, rollout 토큰 열 평균 · 분모=이 행 1행 · 선정=HEADGAP rank 2, generalisation 895행 / 818타깃 중 (기록된 head_gap 0.08 = 분석 head 3개 절대오차의 max − min, 그 두 head는 evoCas9−SpCas9)
"""

GOLDEN_REPORT = """\
9-Hydra v3 (SpCas9) 요약 — 저장된 산출물, 분석 head SpCas9, SpCas9-NG, evoCas9 (primary=SpCas9)
방법별 모집단: ISM 4행 · IG 2행(층당 1행 고정 표본) · rollout 4행

이 데모가 여는 사례: generalisation 층 3개 (A1 ERRLO · B1 ERRHI · C2 HEADGAP)
  선정 모집단 = generalisation 895행 / 818 타깃 — `cases.json`이 적은 수이고, 아래 일치도의 모집단과 다른 모집단이다

성능 — 예측한 head 순서가 측정된 순서와 뒤집힌 행 (kendall_tau_a; 회귀 성능 수치는 이 번들에 저장돼 있지 않다)
  generalisation  13/895행 (1.5%)  그중 복제 노이즈 바닥을 넘는 행 1
  pam_library     169/1,287행 (13.1%)  그중 복제 노이즈 바닥을 넘는 행 0  — 기록된 한정(원문): observed on 5 targets
  바닥 = head별 median, 풀링 없음 (SpCas9 0.0637, SpCas9-NG 0.0697, evoCas9 0.0099)

IG–ISM 일치도 (행별 spearman_rank_correlation의 median, 판정 없음)
  generalisation  0.7  분모 1행 / 1 타깃  (head=SpCas9)
  pam_library     0.45  분모 1행 / 1 타깃  (head=SpCas9)
  층은 합치지 않는다 (pooling 금지). 다른 분석 head는 값이 다르다 — SpCas9-NG 0.55, evoCas9 0.3 (generalisation 층)

한계
  측정된 값으로 제시한다. 합격·불합격 판정을 붙이지 않는다 — 두 기법은 서로의 정답이 아니다.
  rollout은 head 축이 없다 (ADR 0008) — head별 읽기를 줄 수 없고, 모델 내부 정보 의존성이며 물리적 결합의 근거가 아님
  IG는 2행 고정 표본이다 — 사례 선정 전에 자기 seed로 고정됐고, 사례 3개 중 1개만 그 명단에 든다
  사례 번호(A·B·C)는 오차·head 불일치로 고른 행이다 — 기술적 해석에만 쓰고 모집단 요약으로 읽지 않는다
  여기의 모든 수치는 모델 내부 정보 사용을 기술한다. 물리적 DNA/단백질 상호작용에 대한 주장이 아니다.
"""


def test_the_errlo_screen_matches_its_golden_output(reader):
    assert screen.case_screen(reader, "A1") == GOLDEN_A1


def test_the_headgap_screen_matches_its_golden_output(reader):
    assert screen.case_screen(reader, "C2") == GOLDEN_C2


def test_the_report_matches_its_golden_output(reader):
    assert screen.report(reader) == GOLDEN_REPORT


# --- heads ------------------------------------------------------------------ #


def test_headgap_shows_two_heads_and_the_error_cells_one(reader):
    assert screen.heads_shown(reader, reader.case("GEN-HEADGAP-02")) == (
        "SpCas9",
        "evoCas9",
    )
    for case_id in ("GEN-ERRLO-01", "GEN-ERRHI-01"):
        assert screen.heads_shown(reader, reader.case(case_id)) == ("SpCas9",)


def test_the_second_head_is_the_one_whose_stored_error_is_furthest(reader):
    """Not a fixed head, and not the head_gap's own pair: the stored errors'."""
    case = dict(reader.case("GEN-HEADGAP-02"))
    assert screen.comparison_head(reader, case) == "evoCas9"

    # Move the disagreement to the other head and the screen must follow it.
    case["abs_error_SpCas9-NG"] = case["abs_error_SpCas9"] + 5.0
    assert screen.comparison_head(reader, case) == "SpCas9-NG"


def test_the_two_bars_of_a_headgap_screen_are_not_the_same_head(reader):
    text = screen.case_screen(reader, "C2")
    bars = [line for line in text.splitlines() if line.startswith("ISM   ")]
    assert len(bars) == 2
    assert bars[0] != bars[1]
    assert bars[0].endswith("SpCas9") and bars[1].endswith("evoCas9")


def test_the_recorded_head_gap_is_named_as_its_own_quantity(reader):
    """The pair on screen and the recorded `head_gap` are different quantities."""
    text = screen.case_screen(reader, "C2")
    assert "기록된 head_gap" in text
    assert "분석 head 3개 절대오차의 max − min" in text


# --- IG coverage ------------------------------------------------------------ #


def test_the_eight_rows_outside_the_ig_sample_say_so_in_the_ig_slot(reader):
    for number in ("A1", "B1"):
        text = screen.case_screen(reader, number)
        assert screen.IG_SAMPLE_NOTE in text
        # The line sits where the values would be, under the IG heading.
        lines = text.splitlines()
        heading = next(i for i, line in enumerate(lines) if line.startswith("IG ("))
        assert screen.IG_SAMPLE_NOTE in lines[heading + 1]


def test_the_row_inside_the_ig_sample_shows_values_not_the_note(reader):
    text = screen.case_screen(reader, "C2")
    assert screen.IG_SAMPLE_NOTE not in text
    assert "pam_plus_right_flank                SpCas9 +11.400" in text


# --- numbers, strata and refusals ------------------------------------------- #


def test_only_generalisation_cases_get_a_number(reader):
    numbered = screen.numbered_cases(reader)
    assert set(numbered) == {"A1", "B1", "C2"}
    assert all(case["stratum"] == "generalisation" for case in numbered.values())
    # The pam_library case keeps its own case_id in the artifact and is simply
    # not offered: this demo shows one stratum (ADR 0012).
    assert "LIB-ERRLO-01" in reader.case_ids()
    assert all(case["case_id"] != "LIB-ERRLO-01" for case in numbered.values())


def test_a_number_range_with_a_gap_lists_its_numbers_rather_than_a_span(reader):
    assert screen.number_range(reader) == "A1 ERRLO · B1 ERRHI · C2 HEADGAP"


@pytest.mark.parametrize("typed", ["C1", "A9", "Z1"])
def test_an_unknown_number_is_refused_with_the_valid_range(reader, typed):
    with pytest.raises(screen.UnknownCaseNumber) as refusal:
        screen.case_screen(reader, typed)
    assert screen.number_range(reader) in str(refusal.value)


def test_a_number_is_case_insensitive(reader):
    assert screen.case_screen(reader, "c2") == screen.case_screen(reader, "C2")


def test_a_raw_sequence_is_refused_while_its_input_is_unbuilt(reader):
    with pytest.raises(screen.UnknownCaseNumber) as refusal:
        screen.screen_for(reader, "A" * 30)
    assert "30bp" in str(refusal.value)


# --- the guards the golden text rests on ------------------------------------ #


def test_a_label_read_as_percent_is_refused_rather_than_drawn(reader, tmp_path):
    """The split files are percent and `cases.json` fractions; they must agree."""
    case = dict(reader.case("GEN-ERRLO-01"))
    case["label_SpCas9"] = float(case["label_SpCas9"]) * 100
    with pytest.raises(rd.ArtifactMismatch) as refusal:
        screen._checked_row(reader, case, ("SpCas9",))
    assert "label_SpCas9" in str(refusal.value)


def test_the_golden_values_are_the_shown_rows_own(reader):
    """Guards the golden strings themselves against the code that produced them.

    The IG and rollout numbers above are read back here from the stored
    arrays by `row_index`, independently of the screen's own lookups.
    """
    ig = np.load(reader.root / "v3_ig/ig.npz")
    at = int(np.flatnonzero(ig["row_index"] == 11)[0])
    head = fx.IG_HEADS.index("SpCas9")
    assert ig["attribution"][at, head].tolist() == pytest.approx(
        [11.0, 11.1, 11.2, 11.3, 11.4], rel=1e-5
    )

    rollout = np.load(reader.root / "v3_rollout/rollout.npz")
    at = int(np.flatnonzero(rollout["row_index"] == 11)[0])
    assert rollout["rollout"][at].mean(axis=0).tolist() == pytest.approx(
        [35.0, 36.0, 37.0, 38.0, 39.0]
    )


def test_the_sequence_and_the_bar_are_the_same_thirty_columns(reader):
    text = screen.case_screen(reader, "A1")
    lines = {line.split("  ", 1)[0]: line for line in text.splitlines() if "  " in line}
    sequence = lines["서열"].removeprefix("서열  ")
    bar = lines["ISM"].removeprefix("ISM   ").removesuffix("  SpCas9")
    assert len(sequence) == len(bar) == fx.POSITIONS == 30


def test_each_bar_carries_its_own_peak_figure(reader):
    """Separately scaled bars must each state their own maximum.

    Both bars are drawn to their own max, so two full blocks can stand for
    very different magnitudes (on the real C1: 0.454 against 0.035). The
    per-head peak line is the only thing that distinguishes them, so a screen
    that printed one head's peak for both would be caught here by value.
    """
    text = screen.case_screen(reader, "C2")
    peaks = next(
        line for line in text.splitlines() if line.strip().startswith("SpCas9 최대")
    )

    assert "SpCas9 최대 0.900" in peaks
    assert "evoCas9 최대 0.600" in peaks
    # The two bars both reach a full block despite the different magnitudes:
    # that is exactly why the figures above have to be there.
    bars = [line for line in text.splitlines() if line.startswith("ISM   ")]
    assert all("█" in bar for bar in bars)


def test_a_single_head_screen_still_names_whose_peak_it_is(reader):
    text = screen.case_screen(reader, "A1")
    assert "SpCas9 최대 0.900 @ 위치 3" in text


def test_the_rollout_line_says_it_has_no_head_axis(reader):
    """One rollout row stands under two head names, so it must say so."""
    for number in ("A1", "C2"):
        text = screen.case_screen(reader, number)
        rollout = next(
            line for line in text.splitlines() if line.startswith("Attention Rollout")
        )
        assert "ADR 0008" in rollout
        assert "head별 값이 아니다" in rollout


def test_the_recorded_gap_names_its_own_pair_not_the_shown_one(reader):
    """On GEN-HEADGAP-01 the gap's pair excludes the primary head entirely.

    The fixture's own HEADGAP case has evoCas9 furthest from the primary, so
    here the two coincide; the guard is that the pair is stated at all, and
    `gap_pair` is checked directly against a case whose widest pair is not
    the shown one.
    """
    case = dict(reader.case("GEN-HEADGAP-02"))
    assert screen.gap_pair(reader, case) == ("evoCas9", "SpCas9")

    # Give the middle head the extreme error: the gap's pair moves, the shown
    # comparison head does not have to.
    case["abs_error_SpCas9-NG"] = 9.0
    assert screen.gap_pair(reader, case) == ("SpCas9-NG", "SpCas9")
    assert screen.comparison_head(reader, case) == "SpCas9-NG"
