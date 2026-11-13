"""NT-Cas12a 화면 한 장: 사례 번호 하나 → 그 서열의 XAI 결과 (#169, #171).

A case number (`A1`-`A5` low_error, `B1`-`B5` high_error, `C1`-`C5`
out_of_range) is a typing convenience over the audited selection in
`cases.json`: `A3` is the low_error case recorded at `rank_in_category` 3.
Nothing is re-ranked here.

Every figure on the screen is one PAM stratum's, one head's, one row's, so the
four fields `render.Figure` prints per value are gathered into one provenance
line at the bottom instead of being repeated three times per value (#169).
Nothing is computed beyond reading: the ISM bar is the stored `raw_delta`'s
per-position max, IG is the stored signed mass, rollout the stored DNA block.

`report` is the short summary: performance, IG-ISM agreement, limits. It keeps
the old report's care - each figure with its PAM stratum, statistic and actual
denominator, and the pooled `all` population marked as pooled.
"""

from __future__ import annotations

from . import render, screen_parts
from .reader import ArtifactMismatch, Cas12aReader
from .screen_parts import (
    LABEL_WIDTH,
    ROLLOUT_LIMIT,
    UnknownCaseNumber,
    spark_bar,
)
from .sequence_lookup import NotASequence, SequenceNotStored

#: The refusal types are re-exported because the launcher catches a screen
#: module's refusals through the module it was handed; `spark_bar` because the
#: golden tests draw bars with it.
__all__ = [
    "NotASequence",
    "SequenceNotStored",
    "UnknownCaseNumber",
    "case_screen",
    "help_lines",
    "number_range",
    "report",
    "screen_for",
    "sequence_screen",
    "spark_bar",
]

#: Letter -> role, in the order the screen lists them. A and B mean the same
#: thing for both nucleases (A = predicted well, B = badly).
CASE_LETTERS: tuple[tuple[str, str], ...] = (
    ("A", "low_error"),
    ("B", "high_error"),
    ("C", "out_of_range"),
)

# --------------------------------------------------------------------------- #
# case numbers
# --------------------------------------------------------------------------- #


def numbered_cases(reader: Cas12aReader) -> dict[str, dict]:
    """`A1` -> the low_error case at rank 1, and so on, from `cases.csv` as is."""
    letter_of = {role: letter for letter, role in CASE_LETTERS}
    numbered: dict[str, dict] = {}
    for case in reader.cases:
        letter = letter_of.get(str(case["role"]))
        if letter is None:
            raise ArtifactMismatch(f"사례 역할 {case['role']!r}에 번호 글자가 없다")
        number = f"{letter}{int(case['rank_in_category'])}"
        if number in numbered:
            raise ArtifactMismatch(
                f"사례 번호 {number}이(가) 두 번 나온다: {case['role']} "
                f"rank_in_category {case['rank_in_category']}"
            )
        numbered[number] = case
    return numbered


def number_range(reader: Cas12aReader) -> str:
    """The valid numbers, by letter: `A1–A5 low_error · B1–B5 high_error · ...`."""
    return screen_parts.number_range(numbered_cases(reader), CASE_LETTERS)


def help_lines(reader: Cas12aReader) -> list[str]:
    """The input lines of the launcher's help, in this nuclease's own units."""
    return [
        f"  <번호>   사례 한 장: {number_range(reader)}",
        (
            f"  <서열>   {reader.sequence_index.unit.name} → 같은 화면 "
            f"(저장된 {reader.sequence_index.stored_rows:,}행에서 찾는다)"
        ),
    ]


def case_for(reader: Cas12aReader, number: str) -> tuple[str, dict]:
    """The case a typed number names, or a refusal that states the valid range."""
    numbered = numbered_cases(reader)
    key = screen_parts.parse_case_number(number)
    if key not in numbered:
        raise UnknownCaseNumber(
            f"없는 사례 번호다: {number.strip()!r}. 유효 번호: {number_range(reader)}"
        )
    return key, numbered[key]


# --------------------------------------------------------------------------- #
# the screen
# --------------------------------------------------------------------------- #


def _pam_line(pam: tuple[int, int]) -> str:
    """Cas12a brackets its recorded `layout.pam` exactly, and names it PAM."""
    return screen_parts.bracket_line(pam, f"PAM {screen_parts.span_text(*pam)}")


def _checked_case_row(reader: Cas12aReader, case: dict) -> int:
    """The case's row, its cases.csv figures checked against predictions.csv."""
    row = int(case["source_row"])
    stored = reader.prediction(row)
    for field in ("label", "raw_prediction", "raw_absolute_error", "pam_stratum"):
        if stored[field] != case[field]:
            raise ArtifactMismatch(
                f"source_row {row}: cases.csv의 {field}({case[field]})가 "
                f"predictions.csv의 값({stored[field]})과 다르다"
            )
    return row


def row_body(reader: Cas12aReader, row: int) -> list[str]:
    """The screen's body for one row: figures, sequence, ISM, IG, rollout.

    Shared by the number and the sequence input, so the two print the same
    body for the same row; only the header and the provenance line differ.
    Everything is read by `source_row` from `predictions.csv` and the arrays.
    """
    stored = reader.prediction(row)
    sequence = reader.sequence(row)
    ism = reader.ism_max_profile(row)
    spans = reader.token_spans
    pam = reader.pam_span
    if not len(sequence) == len(ism) == spans[-1][1]:
        raise ArtifactMismatch(
            f"source_row {row}: 서열 {len(sequence)}bp, ISM {len(ism)}위치, "
            f"토큰 구간 끝 {spans[-1][1]}이 맞지 않는다"
        )
    ig = reader.ig_dna_mass(row)
    rollout = reader.rollout_referenced(row)
    caption = screen_parts.ism_caption(ism, delta="raw Δ")
    lines = [
        (
            # Significant figures, not decimals: low_error cases' errors are
            # ~1e-4 and would all print as 0.000.
            f"실측 {stored['label']:.4g}   예측 {stored['raw_prediction']:.4g}   "
            f"오차 {stored['raw_absolute_error']:.4g}   (raw, fraction)"
        ),
        "",
        f"서열  {sequence}",
        f"ISM   {spark_bar(ism)}",
        _pam_line(pam),
        " " * LABEL_WIDTH + caption,
        "",
        f"IG (DNA 토큰 {len(spans)}개, 위치 순, signed mass)",
    ]
    width = max(len(screen_parts.span_text(*span)) for span in spans)
    bases = max(end - start for start, end in spans)
    for index, ((start, end), value) in enumerate(zip(spans, ig, strict=True), 1):
        lines.append(
            f"  t{index}  {screen_parts.span_text(start, end):<{width}}  "
            f"{sequence[start:end]:<{bases}}  {value:+.3f}"
        )
    lines += [
        "",
        *screen_parts.rollout_block(rollout, queries=len(rollout)),
        "",
    ]
    return lines


def _provenance(reader: Cas12aReader, row: int, how: str) -> str:
    """The one line that carries the four fields every figure above shares."""
    return (
        f"출처: 층={reader.prediction(row)['pam_stratum']} · head={reader.head} · "
        "통계=ISM 최대 |raw Δ|, IG signed mass, rollout DNA 블록 열 평균 · "
        f"분모=이 행 1행 · {how}"
    )


def case_screen(reader: Cas12aReader, number: str) -> str:
    """The one screen for one case number."""
    key, case = case_for(reader, number)
    row = _checked_case_row(reader, case)
    pool = reader.case_pool(str(case["role"]))
    lines = [
        f"{key}  {case['role']}  source_row {row}  {case['pam_stratum']}",
        *row_body(reader, row),
        _provenance(
            reader,
            row,
            f"선정={case['role']} 후보 {pool['available']:,}행 중 rank "
            f"{case['rank_in_category']}",
        ),
    ]
    return "\n".join(lines) + "\n"


def sequence_screen(reader: Cas12aReader, typed: str) -> str:
    """The same screen, opened by a 34bp target window instead of a number.

    The window is looked up over every stored row, not only the 15 cases. A
    window on several rows (cell-line replicates) shows the smallest
    `source_row` with one line saying so; nothing is averaged.
    """
    index = reader.sequence_index
    hit = index.find(typed)
    row = hit.row
    numbers = [
        key
        for key, case in numbered_cases(reader).items()
        if int(case["source_row"]) == row
    ]
    header = f"서열 입력  source_row {row}  {reader.prediction(row)['pam_stratum']}"
    if numbers:
        header += f"  (= 사례 {', '.join(numbers)})"
    lines = [header]
    duplicate = hit.duplicate_line()
    if duplicate:
        lines.append(duplicate)
    lines += [
        *row_body(reader, row),
        _provenance(
            reader,
            row,
            f"입력={index.unit.name}, 저장된 {index.stored_rows:,}행에서 조회",
        ),
    ]
    return "\n".join(lines) + "\n"


def screen_for(reader: Cas12aReader, typed: str) -> str:
    """Route one input: a short input with a digit is a case number, else a sequence.

    A number is a letter and a digit or two (`A1`); a sequence is letters. So
    a mistyped number still gets the number's range in its refusal, and a
    mistyped sequence - a stray digit included - gets the 34bp unit in its own.
    """
    if screen_parts.looks_like_case_number(typed):
        return case_screen(reader, typed)
    return sequence_screen(reader, typed)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

#: Within-group first, the pooled population last and marked as pooled.
REPORT_STRATA = ("TTTV", "TTTT", "all")


def _pooled(reader: Cas12aReader, stratum: str, note: str) -> str:
    return f" — TTTV·TTTT 풀링: {note}" if reader.pools_pam_groups(stratum) else ""


#: `summary.json` coverage keys and their on-screen names, in pipeline order.
COVERAGE_METHODS = (
    ("prediction", "예측"),
    ("ism", "ISM"),
    ("ig", "IG"),
    ("rollout", "rollout"),
)


def _coverage(reader: Cas12aReader) -> str:
    """Each method's actual success over what was requested, from `summary.json`.

    The coverage entries are row-ID lists; anything else means the artifact's
    shape changed, and is refused rather than guessed into a count.
    """
    coverage = reader.summary.get("coverage") or {}
    parts = []
    for key, label in COVERAGE_METHODS:
        entry = coverage.get(key)
        if entry is None:
            parts.append(f"{label} 기록 없음")
            continue
        if not all(isinstance(entry.get(k), list) for k in ("available", "requested")):
            raise render.RenderRefused(f"coverage의 {key}가 행 ID 목록이 아니다")
        parts.append(f"{label} {len(entry['available']):,}/{len(entry['requested']):,}")
    return "성공 coverage (성공/요구 행): " + " · ".join(parts)


def report(reader: Cas12aReader) -> str:
    """The short Cas12a summary: performance, IG-ISM agreement, limits."""
    lines = [
        (
            f"NT-Cas12a 요약 — Test {reader.prediction_rows.rows:,}행, 저장된 산출물, "
            f"head={reader.head}"
        ),
        _coverage(reader),
        "",
        "예측 성능 (raw 예측, 척도 fraction; 분모는 각 PAM 집단의 행 수)",
    ]
    for stratum in REPORT_STRATA:
        entry = reader.metrics_for(stratum)
        lines.append(
            f"  {stratum:<4}  Spearman {render.format_value(entry['spearman'])}  "
            f"MAE {render.format_value(entry['mae'])}  분모 {int(entry['count']):,}행"
            + _pooled(reader, stratum, "Spearman이 군 사이 평균 효율 차이를 포함한다")
        )
    primary = reader.agreement_primary("all")["primary"]
    lines.append(f"IG–ISM 일치도 (행별 Spearman의 median, {primary}, 판정 없음)")
    for stratum in REPORT_STRATA:
        entry = reader.agreement_primary(stratum)
        median = entry["median"]
        value = (
            render.format_value(median)
            if median is not None
            else f"정의되지 않음 — {entry['reason']}"
        )
        lines.append(
            f"  {stratum:<4}  {value}  분모 유효 {entry['n_valid']:,}행 / "
            f"전체 {entry['n_total']:,}행"
            + _pooled(reader, stratum, "두 군의 행을 합친 분포의 median")
        )
    degeneracy = reader.rollout_degeneracy("all")
    lines += [
        "한계",
        f"  {render.NO_VERDICT}",
        (
            "  rollout: DNA query가 거의 동일한 행 "
            f"{degeneracy['near_identical_query_count']:,}/"
            f"{degeneracy['n_accepted']:,}행 — 행 사이 차이를 행의 성질로 읽지 않는다 "
            "(저장·수치 실패가 아니라 해석 제한)"
        ),
        (
            "  rollout은 encoder token attention flow이며 출력 attribution이 아니다 — "
            f"{ROLLOUT_LIMIT}"
        ),
        (
            "  사례 번호(A·B·C)는 오차로 고른 행이다 — 기술적 해석에만 쓰고 "
            "모집단 요약으로 읽지 않는다"
        ),
    ]
    return "\n".join(lines) + "\n"
