"""The Cas12a case screen and short report, against golden output (#171).

The fixture is built so the misreads ADR 0012 names show up in the text:

* **`row_index` join.** ISM, IG and rollout each store their rows in a
  different order, none of which leaves the shown row at its default array
  position, and the shown row carries an ISM spike no other row has. Looking
  a row up by array position instead of `source_row` prints another row's
  bars, IG values and rollout values.
* **WT slot.** One position of the shown row has no valid alternative base,
  and its stored deltas are left nonzero. Only "no valid alternative" (`·`) is
  right there. The WT slot stays valid, as on the real data, so keeping its
  zero in the max prints the lowest bar; ignoring validity prints a bar from
  deltas that do not count.

The golden strings are the whole screen, so any change to what the meeting
sees is a visible diff here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xai_test.nt_xai_demo import cas12a_screen as screen
from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo.test import reader_fixtures as fx

#: The case shown: `A1` is the low_error case at rank 1, source_row 9.
SHOWN_ROW = 9
SPIKE_POSITION = 6


def _spike(cas12a_dir, row: int, position: int, value: float) -> None:
    """Give one row's ISM a peak that no other row has."""
    path = cas12a_dir / "ism.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    at = int(np.flatnonzero(arrays["source_row"] == row)[0])
    alternatives = arrays["non_wt_mask"][at, position]
    arrays["raw_delta"][at, position] = np.where(alternatives, -value, 0.0)
    np.savez(path, **arrays)


@pytest.fixture
def reader(tmp_path):
    data = fx.build_data_dir(tmp_path / "data")
    cas12a_dir = data / "cas12a"
    _spike(cas12a_dir, SHOWN_ROW, SPIKE_POSITION, 0.9)
    fx.reorder_cas12a_arrays(cas12a_dir, fx.CAS12A_SHUFFLED_ORDERS)
    fx.drop_alternatives(cas12a_dir, SHOWN_ROW, 10)
    return rd.Cas12aReader(data)


GOLDEN_A1 = """\
A1  low_error  source_row 9  TTTV
실측 0.3   예측 0.35   오차 0.05   (raw, fraction)

서열  TGCATGCATGCATGCATGCATGCATGCATGCATG
ISM   ▃▃▃▃▃▃█▃▃▃·▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
           [--]  PAM 5-8
      막대 = 위치별 대체염기 3개 중 최대 |raw Δ| (WT 칸 제외) · 이 행 최대 0.900 @ 위치 6 (· = 유효 대체염기 없음)

IG (DNA 토큰 9개, 위치 순, signed mass)
  t1  0      T       +13.000
  t2  1      G       +14.000
  t3  2      C       +15.000
  t4  3      A       +16.000
  t5  4-9    TGCATG  +17.000
  t6  10-15  CATGCA  +18.000
  t7  16-21  TGCATG  +19.000
  t8  22-27  CATGCA  +20.000
  t9  28-33  TGCATG  +21.000

Attention Rollout (토큰별 참조되는 정도, DNA query 9개 평균)
  t1 117.000  t2 118.000  t3 119.000  t4 120.000  t5 121.000  t6 122.000  t7 123.000  t8 124.000  t9 125.000
  모델 내부 정보 의존성이며 물리적 결합의 근거가 아님

출처: 층=TTTV · head=단일 출력 · 통계=ISM 최대 |raw Δ|, IG signed mass, rollout DNA 블록 열 평균 · 분모=이 행 1행 · 선정=low_error 후보 372행 중 rank 1
"""


GOLDEN_REPORT = """\
NT-Cas12a 요약 — Test 3행, 저장된 산출물, head=단일 출력
성공 coverage (성공/요구 행): 예측 3/3 · ISM 3/3 · IG 3/3 · rollout 3/3

예측 성능 (raw 예측, 척도 fraction; 분모는 각 PAM 집단의 행 수)
  TTTV  Spearman 0.5423  MAE 0.17  분모 2행
  TTTT  Spearman 0.2995  MAE 0.17  분모 1행
  all   Spearman 0.7007  MAE 0.17  분모 3행 — TTTV·TTTT 풀링: Spearman이 군 사이 평균 효율 차이를 포함한다
IG–ISM 일치도 (행별 Spearman의 median, five_6mer/all_nonWT/density, 판정 없음)
  TTTV  0.3  분모 유효 2행 / 전체 2행
  TTTT  0.6  분모 유효 1행 / 전체 1행
  all   0.4  분모 유효 3행 / 전체 3행 — TTTV·TTTT 풀링: 두 군의 행을 합친 분포의 median
한계
  측정된 값으로 제시한다. 합격·불합격 판정을 붙이지 않는다 — 두 기법은 서로의 정답이 아니다.
  rollout: DNA query가 거의 동일한 행 3/3행 — 행 사이 차이를 행의 성질로 읽지 않는다 (저장·수치 실패가 아니라 해석 제한)
  rollout은 encoder token attention flow이며 출력 attribution이 아니다 — 모델 내부 정보 의존성이며 물리적 결합의 근거가 아님
  사례 번호(A·B·C)는 오차로 고른 행이다 — 기술적 해석에만 쓰고 모집단 요약으로 읽지 않는다
"""


def test_the_case_screen_matches_its_golden_output(reader):
    assert screen.case_screen(reader, "A1") == GOLDEN_A1


def test_the_report_matches_its_golden_output(reader):
    assert screen.report(reader) == GOLDEN_REPORT


def test_the_golden_values_are_the_shown_rows_own(reader, tmp_path):
    """The golden IG and rollout numbers are row 9's, read independently.

    Guards the golden string itself: it was produced by the code under test,
    so its row-specific numbers are checked against the stored arrays here.
    """
    data = reader.root
    ig = np.load(data / "ig.npz")
    at = int(np.flatnonzero(ig["source_row"] == SHOWN_ROW)[0])
    assert list(ig["signed_mass"][at][1:10]) == [float(v) for v in range(13, 22)]
    rollout = np.load(data / "rollout.npz")
    at = int(np.flatnonzero(rollout["source_row"] == SHOWN_ROW)[0])
    assert rollout["sequence"][at].mean(axis=0)[0] == 117.0


def test_numbers_are_the_selection_ranks_and_not_a_re_ranking(reader):
    numbered = screen.numbered_cases(reader)
    assert {
        key: (case["source_row"], case["role"]) for key, case in numbered.items()
    } == {
        "A1": (9, "low_error"),
        "C1": (4, "out_of_range"),
    }


def test_a_number_is_read_case_insensitively(reader):
    assert screen.case_screen(reader, "a1") == screen.case_screen(reader, "A1")


@pytest.mark.parametrize("typed", ["B1", "A2", "A", "13768", "A1x"])
def test_an_unknown_number_states_the_valid_range(reader, typed):
    with pytest.raises(screen.UnknownCaseNumber) as refusal:
        screen.case_screen(reader, typed)
    assert "유효 번호: A1 low_error · C1 out_of_range" in str(refusal.value)


def test_the_range_is_written_as_spans_on_the_real_shape():
    """Five ranks per letter print as `A1–A5`, as the meeting will see them."""

    five = SimpleNamespace(
        cases=[
            {"role": role, "rank_in_category": rank, "source_row": rank}
            for _, role in screen.CASE_LETTERS
            for rank in range(1, 6)
        ]
    )

    assert screen.number_range(five) == (
        "A1–A5 low_error · B1–B5 high_error · C1–C5 out_of_range"
    )


def test_a_selection_that_repeats_a_number_is_refused():
    twice = SimpleNamespace(
        cases=[
            {"role": "low_error", "rank_in_category": 1, "source_row": 1},
            {"role": "low_error", "rank_in_category": 1, "source_row": 2},
        ]
    )

    with pytest.raises(rd.ArtifactMismatch):
        screen.numbered_cases(twice)


def test_spark_bar_scales_to_the_rows_own_max_and_marks_missing():
    assert screen.spark_bar([0.0, 0.5, 1.0, float("nan")]) == "▁▅█·"


# --------------------------------------------------------------------------- #
# raw 34bp sequence input (#172)
# --------------------------------------------------------------------------- #

#: Row 9's 34bp window in the fixture, i.e. case A1's.
A1_SEQUENCE = "TGCATGCATGCATGCATGCATGCATGCATGCATG"


def _body(screen_text: str) -> list[str]:
    """The screen between its header and its provenance line."""
    lines = screen_text.rstrip("\n").split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("실측 "))
    assert lines[-1].startswith("출처: ")
    return lines[start:-1]


GOLDEN_SEQUENCE_HEAD = (
    "서열 입력  source_row 9  TTTV  (= 사례 A1)\n"
    + GOLDEN_A1.split("\n", 1)[1].rsplit("출처: ", 1)[0]
    + "출처: 층=TTTV · head=단일 출력 · 통계=ISM 최대 |raw Δ|, IG signed mass, "
    "rollout DNA 블록 열 평균 · 분모=이 행 1행 · 입력=34bp 타깃 서열, "
    "저장된 3행에서 조회\n"
)


def test_a_stored_sequence_opens_its_rows_screen(reader):
    assert screen.sequence_screen(reader, A1_SEQUENCE) == GOLDEN_SEQUENCE_HEAD


def test_a_number_and_its_sequence_print_the_same_body(reader):
    by_number = screen.case_screen(reader, "A1")
    by_sequence = screen.sequence_screen(reader, A1_SEQUENCE.lower())
    assert _body(by_number) == _body(by_sequence)
    assert len(_body(by_number)) > 10


def test_a_row_that_is_no_case_gets_no_case_number(reader):
    sequence = reader.prediction(21)["sequence"]
    text = screen.sequence_screen(reader, sequence)
    assert text.startswith("서열 입력  source_row 21  TTTV\n")
    assert "사례" not in text.split("\n")[0]


@pytest.fixture
def replicated(tmp_path):
    """Row 21 carries row 9's window, as a second cell-line measurement would."""
    data = fx.build_data_dir(tmp_path / "data")
    fx.copy_sequence(data / "cas12a", 9, 21)
    fx.reorder_cas12a_arrays(data / "cas12a", fx.CAS12A_SHUFFLED_ORDERS)
    return rd.Cas12aReader(data)


def test_a_replicated_sequence_shows_the_smallest_row_with_one_line(replicated):
    lines = screen.sequence_screen(replicated, A1_SEQUENCE).split("\n")
    assert lines[0] == "서열 입력  source_row 9  TTTV  (= 사례 A1)"
    assert lines[1] == (
        "이 타깃은 2행 중복(세포주 반복): source_row 9, 21 — "
        "가장 작은 source_row 9를 보여준다"
    )
    assert lines[2].startswith("실측 ")


def test_a_replicated_sequence_keeps_the_number_screens_body(replicated):
    assert _body(screen.sequence_screen(replicated, A1_SEQUENCE)) == _body(
        screen.case_screen(replicated, "A1")
    )


def test_a_sequence_not_stored_gives_the_count_and_examples(reader):
    with pytest.raises(screen.SequenceNotStored) as miss:
        screen.sequence_screen(reader, "A" * 34)
    assert str(miss.value) == (
        f"저장된 3행에 없음: {'A' * 34}. 예: "
        "ATGCATGCATGCATGCATGCATGCATGCATGCAT, "
        "TGCATGCATGCATGCATGCATGCATGCATGCATG, "
        "GCATGCATGCATGCATGCATGCATGCATGCATGC"
    )


@pytest.mark.parametrize(
    ("typed", "reason"),
    [
        (A1_SEQUENCE[:-1], "길이 33 (34이어야 한다)"),
        ("ACGU" + A1_SEQUENCE[4:], "ACGT 외 문자 'U' (위치 3)"),
        (A1_SEQUENCE[:10] + "1" + A1_SEQUENCE[11:], "ACGT 외 문자 '1' (위치 10)"),
    ],
)
def test_a_wrong_unit_names_the_34bp_unit(reader, typed, reason):
    with pytest.raises(screen.NotASequence) as refusal:
        screen.screen_for(reader, typed)
    assert str(refusal.value) == f"34bp 타깃 서열이 아니다: {reason}"


@pytest.mark.parametrize(
    ("typed", "route"),
    [
        ("A1", "case_screen"),
        ("c1", "case_screen"),
        ("B3", "case_screen"),
        (A1_SEQUENCE, "sequence_screen"),
        ("ACGT", "sequence_screen"),
        (A1_SEQUENCE[:10] + "1" + A1_SEQUENCE[11:], "sequence_screen"),
        ("A12", "case_screen"),
    ],
)
def test_a_short_input_with_a_digit_is_a_number_and_else_a_sequence(
    reader, monkeypatch, typed, route
):
    taken: list[str] = []
    for name in ("case_screen", "sequence_screen"):
        monkeypatch.setattr(
            screen, name, lambda _reader, _typed, name=name: taken.append(name)
        )
    screen.screen_for(reader, typed)
    assert taken == [route]
