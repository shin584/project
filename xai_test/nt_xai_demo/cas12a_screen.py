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

import math
import re
from collections.abc import Sequence

from . import render
from .reader import ArtifactMismatch, Cas12aReader

#: Letter -> role, in the order the screen lists them. A and B mean the same
#: thing for both nucleases (A = predicted well, B = badly).
CASE_LETTERS: tuple[tuple[str, str], ...] = (
    ("A", "low_error"),
    ("B", "high_error"),
    ("C", "out_of_range"),
)
_NUMBER = re.compile(r"([A-Za-z])(\d+)")

#: Eight heights for the ISM bar; a missing value is a dot, never a low bar.
SPARKS = "▁▂▃▄▅▆▇█"
MISSING = "·"

ROLLOUT_LIMIT = "모델 내부 정보 의존성이며 물리적 결합의 근거가 아님"
LABEL_WIDTH = 6  # "서열  " / "ISM   " in terminal columns


class UnknownCaseNumber(KeyError):
    """A case number the selection does not hold. Its message names the range."""

    def __str__(self) -> str:  # KeyError would quote the message
        return str(self.args[0])


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
    numbered = numbered_cases(reader)
    parts = []
    for letter, role in CASE_LETTERS:
        ranks = sorted(int(n[1:]) for n in numbered if n[0] == letter)
        if not ranks:
            continue
        if ranks == list(range(ranks[0], ranks[-1] + 1)) and len(ranks) > 1:
            span = f"{letter}{ranks[0]}–{letter}{ranks[-1]}"
        else:
            span = ", ".join(f"{letter}{rank}" for rank in ranks)
        parts.append(f"{span} {role}")
    return " · ".join(parts)


def case_for(reader: Cas12aReader, number: str) -> tuple[str, dict]:
    """The case a typed number names, or a refusal that states the valid range."""
    match = _NUMBER.fullmatch(number.strip())
    numbered = numbered_cases(reader)
    key = f"{match.group(1).upper()}{int(match.group(2))}" if match else None
    if key not in numbered:
        raise UnknownCaseNumber(
            f"없는 사례 번호다: {number.strip()!r}. 유효 번호: {number_range(reader)}"
        )
    return key, numbered[key]


# --------------------------------------------------------------------------- #
# the screen
# --------------------------------------------------------------------------- #


def spark_bar(values: Sequence[float]) -> str:
    """One character per position, scaled to the row's own max; NaN is a dot."""
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        raise render.RenderRefused("ISM 막대에 그릴 값이 하나도 없다")
    peak = max(finite)
    top = len(SPARKS) - 1
    return "".join(
        MISSING if math.isnan(v) else SPARKS[round(v / peak * top) if peak > 0 else 0]
        for v in values
    )


def _span_text(start: int, end: int) -> str:
    """A half-open span as the inclusive positions a reader counts."""
    return f"{start}" if end - start == 1 else f"{start}-{end - 1}"


def _pam_line(pam: tuple[int, int]) -> str:
    start, end = pam
    width = end - start
    bracket = "[" + "-" * max(width - 2, 0) + "]" if width > 1 else "^"
    return " " * (LABEL_WIDTH + start) + bracket + f"  PAM {_span_text(start, end)}"


def case_screen(reader: Cas12aReader, number: str) -> str:
    """The one screen for one case number."""
    key, case = case_for(reader, number)
    row = int(case["source_row"])
    stored = reader.prediction(row)
    for field in ("label", "raw_prediction"):
        if stored[field] != case[field]:
            raise ArtifactMismatch(
                f"source_row {row}: cases.csv의 {field}({case[field]})가 "
                f"predictions.csv의 값({stored[field]})과 다르다"
            )
    stratum = str(case["pam_stratum"])
    if stored["pam_stratum"] != stratum:
        raise ArtifactMismatch(
            f"source_row {row}: PAM 집단이 cases.csv {stratum}, "
            f"predictions.csv {stored['pam_stratum']}로 다르다"
        )

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
    peak = render.profile_peak(ism)
    pool = reader.case_pool(str(case["role"]))

    caption = (
        "막대 = 위치별 대체염기 3개 중 최대 |raw Δ| (WT 칸 제외) · "
        f"이 행 최대 {ism[peak]:.3f} @ 위치 {peak}"
    )
    if any(math.isnan(v) for v in ism):
        caption += f" ({MISSING} = 유효 대체염기 없음)"
    lines = [
        f"{key}  {case['role']}  source_row {row}  {stratum}",
        (
            # Significant figures, not decimals: low_error cases' errors are
            # ~1e-4 and would all print as 0.000.
            f"실측 {case['label']:.4g}   예측 {case['raw_prediction']:.4g}   "
            f"오차 {case['raw_absolute_error']:.4g}   (raw, fraction)"
        ),
        "",
        f"서열  {sequence}",
        f"ISM   {spark_bar(ism)}",
        _pam_line(pam),
        " " * LABEL_WIDTH + caption,
        "",
        f"IG (DNA 토큰 {len(spans)}개, 위치 순, signed mass)",
    ]
    width = max(len(_span_text(*span)) for span in spans)
    bases = max(end - start for start, end in spans)
    for index, ((start, end), value) in enumerate(zip(spans, ig, strict=True), 1):
        lines.append(
            f"  t{index}  {_span_text(start, end):<{width}}  "
            f"{sequence[start:end]:<{bases}}  {value:+.3f}"
        )
    lines += [
        "",
        f"Attention Rollout (토큰별 참조되는 정도, DNA query {len(rollout)}개 평균)",
        "  "
        + "  ".join(f"t{index} {value:.3f}" for index, value in enumerate(rollout, 1)),
        f"  {ROLLOUT_LIMIT}",
        "",
        (
            f"출처: 층={stratum} · head={reader.head} · 통계=ISM 최대 |raw Δ|, "
            "IG signed mass, rollout DNA 블록 열 평균 · 분모=이 행 1행 · 선정="
            f"{case['role']} 후보 {pool['available']:,}행 중 rank "
            f"{case['rank_in_category']}"
        ),
    ]
    return "\n".join(lines) + "\n"


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
