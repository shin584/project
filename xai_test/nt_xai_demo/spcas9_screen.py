"""9-Hydra v3 화면 한 장: 사례 번호 하나 → 그 서열의 XAI 결과 (#169, #173).

A case number (`A1`-`A3` ERRLO, `B1`-`B3` ERRHI, `C1`-`C3` HEADGAP) is a
typing convenience over the audited selection in `cases.json`: `B2` is the
ERRHI case recorded at `selection_rank` 2. Nothing is re-ranked here.

Only the **generalisation** stratum is offered. The nine `pam_library` cases
are variants of 5 targets and their selection bounds are their own (the two
strata's gap scales differ about twofold), so with one stratum on screen there
is no path left that could mix them - ADR 0012's prohibition made structural
rather than restated in a caption.

Three things this nuclease's artifacts force, which Cas12a's do not:

* **Heads.** ERRLO and ERRHI rank on the primary head's own absolute error, so
  one head is shown. HEADGAP was selected *because* the heads disagree, so a
  second head is shown beside the primary: the analysed head whose stored
  absolute error sits furthest from the primary's. Showing one head there
  would take the selection's reason off the screen.
* **IG covers 400 rows, not all of them.** Of the nine cases exactly one
  (`GEN-HEADGAP-02`) is in the frozen subset. The other eight print one line
  saying the sample does not cover the row - an absence with its reason, not a
  blank that reads as a lost number.
* **No recorded PAM coordinates.** `ig.sidecar.json` records token boundaries
  and names, and the last token (24-29) is `pam_plus_right_flank`. That span
  is bracketed under its recorded name; the PAM's own 3 bases are not
  separated out, because nothing in these artifacts says where they sit.

Every figure is one stratum's, one head's, one row's, so `render.Figure`'s four
fields are gathered into one provenance line at the bottom. New computation is
zero: the bar, the token values and the rollout are read from stored arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import render, screen_parts
from .reader import ArtifactMismatch, SpCas9Reader
from .screen_parts import ROLLOUT_LIMIT, UnknownCaseNumber

__all__ = [
    "UnknownCaseNumber",
    "case_screen",
    "help_lines",
    "number_range",
    "numbered_cases",
    "report",
    "screen_for",
]

#: Letter -> criterion, in the order the screen lists them. A and B mean the
#: same thing for both nucleases (A = predicted well, B = badly); C is this
#: nuclease's own third group.
CASE_LETTERS: tuple[tuple[str, str], ...] = (
    ("A", "ERRLO"),
    ("B", "ERRHI"),
    ("C", "HEADGAP"),
)
#: The one stratum the demo offers. `pam_library` stays out (see the module
#: docstring); its cases remain in the artifact and keep their own `case_id`.
STRATUM = "generalisation"
HEADGAP = "HEADGAP"

#: The sample line the eight cases outside the frozen IG subset print.
IG_SAMPLE_NOTE = "IG는 층당 200행 표본 — 이 행은 비대상"

#: How close the case table's own figures must sit to the split and prediction
#: tables' before the screen trusts the join. They are the same quantity
#: written twice: `cases.json` rounds to 15 digits and the arrays are float32,
#: so they agree to about 1e-7 and never exactly.
JOIN_TOLERANCE = 1e-6


def _case_number(case: dict[str, Any]) -> str:
    letter_of = {criterion: letter for letter, criterion in CASE_LETTERS}
    criterion = str(case["criterion"])
    letter = letter_of.get(criterion)
    if letter is None:
        raise ArtifactMismatch(f"사례 기준 {criterion!r}에 번호 글자가 없다")
    return f"{letter}{int(case['selection_rank'])}"


def numbered_cases(reader: SpCas9Reader) -> dict[str, dict[str, Any]]:
    """`A1` -> the ERRLO case at selection rank 1, from `cases.json` as is.

    Generalisation only. The numbering is a label over the audited selection:
    the rank is the selection's own `selection_rank`, not a position in any
    list this module builds.
    """
    numbered: dict[str, dict[str, Any]] = {}
    for case in reader.cases:
        if str(case["stratum"]) != STRATUM:
            continue
        number = _case_number(case)
        if number in numbered:
            raise ArtifactMismatch(
                f"사례 번호 {number}이(가) 두 번 나온다: {case['case_id']}와 "
                f"{numbered[number]['case_id']}"
            )
        numbered[number] = dict(case)
    if not numbered:
        raise ArtifactMismatch(f"{STRATUM} 층의 사례가 하나도 없다")
    return numbered


def number_range(reader: SpCas9Reader) -> str:
    """The valid numbers, by letter: `A1–A3 ERRLO · B1–B3 ERRHI · ...`."""
    return screen_parts.number_range(numbered_cases(reader), CASE_LETTERS)


def help_lines(reader: SpCas9Reader) -> list[str]:
    """The input lines of the launcher's help, in this nuclease's own units.

    The 30bp sequence input lands in #174, so it is not offered here: a help
    text that listed an input the session refuses would be the meeting's
    first wrong turn.
    """
    return [
        f"  <번호>   사례 한 장: {number_range(reader)}",
        (
            f"           {STRATUM} 층만 — 서열은 ISM이 덮는 30bp 창 "
            "(20bp protospacer가 그 안에 든다)"
        ),
    ]


def case_for(reader: SpCas9Reader, number: str) -> tuple[str, dict[str, Any]]:
    """The case a typed number names, or a refusal that states the valid range."""
    numbered = numbered_cases(reader)
    key = screen_parts.parse_case_number(number)
    if key not in numbered:
        raise UnknownCaseNumber(
            f"없는 사례 번호다: {number.strip()!r}. 유효 번호: {number_range(reader)}"
        )
    return key, numbered[key]


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #


def _recorded_error(case: dict[str, Any], head: str) -> float:
    """The case table's own recorded absolute error for one head."""
    value = case.get(f"abs_error_{head}")
    if value is None:
        raise ArtifactMismatch(
            f"사례 {case['case_id']}에 head {head}의 abs_error가 없다"
        )
    return float(value)


def comparison_head(reader: SpCas9Reader, case: dict[str, Any]) -> str:
    """The analysed head whose recorded error sits furthest from the primary's.

    HEADGAP selected the row for cross-head disagreement, so the second head
    on screen is the one that disagreement is widest against. This is a choice
    among stored values, not a new statistic - and it is deliberately *not*
    the recorded `head_gap`, which is max - min over all three heads and can
    be a different pair entirely. The provenance line quotes that recorded gap
    under its own name.
    """
    primary = reader.primary_head
    others = [head for head in reader.analysed_heads if head != primary]
    if not others:
        raise ArtifactMismatch("비교할 다른 분석 head가 없다")
    return max(
        others,
        key=lambda head: abs(
            _recorded_error(case, head) - _recorded_error(case, primary)
        ),
    )


def gap_pair(reader: SpCas9Reader, case: dict[str, Any]) -> tuple[str, str]:
    """The two heads the recorded `head_gap` is the distance between.

    `head_gap` is max - min over all three analysed heads, so its pair need
    not include the primary head and often does not: on `GEN-HEADGAP-01` the
    gap runs SpCas9-NG to evoCas9 while the screen shows SpCas9 and evoCas9.
    Naming it stops the quoted gap reading as the difference between the two
    heads drawn above it.
    """
    errors = {head: _recorded_error(case, head) for head in reader.analysed_heads}
    return max(errors, key=errors.get), min(errors, key=errors.get)


def heads_shown(reader: SpCas9Reader, case: dict[str, Any]) -> tuple[str, ...]:
    """One head, or two when the selection reason is head disagreement."""
    primary = reader.primary_head
    if str(case["criterion"]) != HEADGAP:
        return (primary,)
    return (primary, comparison_head(reader, case))


# --------------------------------------------------------------------------- #
# the screen
# --------------------------------------------------------------------------- #


def _checked_row(
    reader: SpCas9Reader, case: dict[str, Any], heads: Sequence[str]
) -> int:
    """The case's row, its recorded figures checked against the split tables.

    `cases.json` stores labels and predictions as fractions; the split files
    store labels as percent. Reading one as the other is the trap this guard
    closes, and it closes it on every head the screen is about to print.
    """
    row = int(case["row_index"])
    for head in heads:
        for field, stored in (
            (f"label_{head}", reader.label(row, head)),
            (f"pred_{head}", reader.prediction(row, head)),
        ):
            recorded = case.get(field)
            if recorded is None or stored is None:
                raise ArtifactMismatch(
                    f"row_index {row}: {field}이(가) 한쪽에만 있다 "
                    f"(cases.json={recorded}, 표={stored})"
                )
            if abs(float(recorded) - float(stored)) > JOIN_TOLERANCE:
                raise ArtifactMismatch(
                    f"row_index {row}: cases.json의 {field}({recorded})가 "
                    f"split·예측 표의 값({stored})과 다르다"
                )
    return row


def _figures_line(reader: SpCas9Reader, row: int, head: str) -> str:
    """One head's measured, predicted and absolute error, all fractions."""
    label = reader.label(row, head)
    prediction = reader.prediction(row, head)
    return (
        # Significant figures, not decimals: an ERRLO case's error is ~1e-5
        # and would otherwise print as 0.000.
        f"  {head:<10} 실측 {label:.4g}   예측 {prediction:.4g}   "
        f"오차 {abs(label - prediction):.4g}"
    )


def _ig_block(reader: SpCas9Reader, row: int, heads: Sequence[str]) -> list[str]:
    """IG on the five tokens, or the one line saying the sample misses this row.

    The absence is printed where the values would have been, in the same
    place for all eight rows it applies to, so "why does SpCas9 have no IG
    here" is answered on screen instead of looking like a gap.
    """
    names = reader.token_names
    boundaries = reader.token_boundaries
    if not reader.has_ig(row):
        return [
            f"IG (토큰 {len(names)}개, 위치 순, signed)",
            (
                f"  {IG_SAMPLE_NOTE} (고정 명단 {reader.ig_rows.rows:,}행, "
                "사례 선정 전에 자기 seed로 고정)"
            ),
        ]
    lines = [
        f"IG (토큰 {len(names)}개, 위치 순, signed, {reader.ig_scale})",
    ]
    width = max(
        len(screen_parts.span_text(start, end + 1)) for start, end in boundaries
    )
    name_width = max(len(name) for name in names)
    values = {head: reader.ig_tokens(row, head) for head in heads}
    for index, ((start, end), name) in enumerate(
        zip(boundaries, names, strict=True), 1
    ):
        shown = "  ".join(f"{head} {values[head][index - 1]:+.3f}" for head in heads)
        lines.append(
            f"  t{index}  {screen_parts.span_text(start, end + 1):<{width}}  "
            f"{name:<{name_width}}  {shown}"
        )
    return lines


def row_body(reader: SpCas9Reader, row: int, heads: Sequence[str]) -> list[str]:
    """The screen's body for one row: figures, sequence, ISM, IG, rollout.

    The sequence is the 30bp window ISM covers, with the 20bp protospacer
    sitting inside it, so the letters line up with the 30-column bar. One bar
    per head shown, and **each bar carries its own peak figure**: the bars are
    scaled to their own maxima (the analyses' own convention, so that a head's
    smaller dynamic range is not drawn as a difference in where it reads), and
    the heads differ several-fold, so two equal-looking bars can stand for very
    different magnitudes. Without the per-head peak that would read as two
    heads being equally sensitive.
    """
    sequence = reader.reference_sequence(row)
    pam_token = len(reader.token_boundaries) - 1
    pam_span = screen_parts.inclusive_span(*reader.token_boundaries[pam_token])
    lines = [_figures_line(reader, row, head) for head in heads]
    lines += ["", f"서열  {sequence}"]
    profiles = {}
    for head in heads:
        profile = reader.ism_max_profile(row, head)
        if len(sequence) != len(profile):
            raise ArtifactMismatch(
                f"row_index {row}: 서열 {len(sequence)}bp와 ISM {len(profile)}위치가 "
                "맞지 않는다"
            )
        profiles[head] = profile
        lines.append(f"ISM   {screen_parts.spark_bar(profile)}  {head}")
    # The last token's recorded name, not "PAM": these artifacts record no PAM
    # coordinates, and the token binds the PAM to the right flank.
    lines.append(
        screen_parts.bracket_line(
            pam_span,
            f"{reader.token_names[pam_token]} {screen_parts.span_text(*pam_span)}",
        )
    )
    indent = " " * screen_parts.LABEL_WIDTH
    lines.append(
        indent
        + screen_parts.bar_definition()
        + " · 각 막대는 자기 최대에 맞춰 그린다 (fraction)"
        + screen_parts.missing_note(list(profiles.values()))
    )
    # Each bar's own peak, because the bars are separately scaled: without
    # these, two equal-looking bars would read as equal sensitivity.
    lines.append(
        indent
        + " · ".join(screen_parts.peak_text(profiles[head], head) for head in heads)
    )
    lines += ["", *_ig_block(reader, row, heads)]
    rollout = reader.rollout(row)
    lines += [
        "",
        *screen_parts.rollout_block(
            reader.rollout_referenced(row),
            # The query count is the matrix's rows, not the values returned.
            queries=len(rollout),
            query_name="토큰",
            head_note=screen_parts.ROLLOUT_HEAD_INVARIANCE,
        ),
        "",
    ]
    return lines


def _provenance(
    reader: SpCas9Reader, case: dict[str, Any], heads: Sequence[str]
) -> str:
    """The one line carrying the four fields every figure above shares."""
    pool = reader.case_strata[STRATUM]
    gap = case.get("head_gap")
    selection = (
        f"선정={case['criterion']} rank {case['selection_rank']}, "
        f"{STRATUM} {int(pool['rows']):,}행 / "
        f"{int(pool['distinct_protospacers']):,}타깃 중"
    )
    if str(case["criterion"]) == HEADGAP and gap is not None:
        widest, narrowest = gap_pair(reader, case)
        selection += (
            f" (기록된 head_gap {float(gap):.4g} = 분석 head 3개 절대오차의 "
            f"max − min, 그 두 head는 {widest}−{narrowest})"
        )
    return (
        f"출처: 층={STRATUM} · head={', '.join(heads)} · "
        "통계=ISM 최대 |Δ|(fraction), IG signed token attribution, "
        f"rollout 토큰 열 평균 · 분모=이 행 1행 · {selection}"
    )


def case_screen(reader: SpCas9Reader, number: str) -> str:
    """The one screen for one case number."""
    key, case = case_for(reader, number)
    heads = heads_shown(reader, case)
    row = _checked_row(reader, case, heads)
    header = (
        f"{key}  {case['case_id']}  {case['criterion']}  row_index {row}  {STRATUM}"
    )
    if len(heads) > 1:
        header += f"  (head 2개: {heads[1]}가 {heads[0]} 대비 오차 차이 최대)"
    lines = [header, *row_body(reader, row, heads), _provenance(reader, case, heads)]
    return "\n".join(lines) + "\n"


def screen_for(reader: SpCas9Reader, typed: str) -> str:
    """Route one input. The raw 30bp sequence input lands in #174.

    Until it does, anything that is not a case number is refused by the same
    refusal a wrong number gets, naming the numbers that do work: the screen
    these artifacts can open today is the numbered one.
    """
    if screen_parts.looks_like_case_number(typed):
        return case_screen(reader, typed)
    raise UnknownCaseNumber(
        f"사례 번호가 아니다: {typed.strip()[:40]!r}. "
        f"유효 번호: {number_range(reader)} "
        "(30bp 생서열로 여는 입력은 아직 없다)"
    )


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def report(reader: SpCas9Reader) -> str:
    """The short 9-Hydra v3 summary: performance, IG-ISM agreement, limits.

    Two populations appear and they are different populations, so each is
    written with its own denominator and they are never added: the selection's
    own `generalisation` pool as `cases.json` records it, and the IG-ISM
    agreement's frozen 200-row-per-stratum subset. `ig_ism_validation.json`
    declares `"pooling": "forbidden"`, so no figure here spans both strata.
    """
    primary = reader.primary_head
    pool = reader.case_strata[STRATUM]
    numbered = numbered_cases(reader)
    lines = [
        (
            f"9-Hydra v3 (SpCas9) 요약 — 저장된 산출물, 분석 head "
            f"{', '.join(reader.analysed_heads)} (primary={primary})"
        ),
        (
            f"방법별 모집단: ISM {reader.ism_rows.rows:,}행 · "
            f"IG {reader.ig_rows.rows:,}행(층당 "
            f"{int(reader.ig_subset['rows_per_stratum']):,}행 고정 표본) · "
            f"rollout {reader.rollout_rows.rows:,}행"
        ),
        "",
        (
            f"이 데모가 여는 사례: {STRATUM} 층 {len(numbered)}개 "
            f"({number_range(reader)})"
        ),
        (
            f"  선정 모집단 = {STRATUM} {int(pool['rows']):,}행 / "
            f"{int(pool['distinct_protospacers']):,} 타깃 — `cases.json`이 적은 수이고, "
            "아래 일치도의 모집단과 다른 모집단이다"
        ),
        "",
        (
            "성능 — 예측한 head 순서가 측정된 순서와 뒤집힌 행 "
            f"({reader.head_order_reversal['statistic']}; 회귀 성능 수치는 이 번들에 "
            "저장돼 있지 않다)"
        ),
    ]
    for stratum in reader.strata:
        entry = reader.reversal_stratum(stratum)
        clearing = entry["clearing_floor"]
        line = (
            f"  {stratum:<15} {int(entry['reversal_rows']):,}/"
            f"{int(entry['rows']):,}행 ({float(entry['reversal_rate']):.1%})  "
            f"그중 복제 노이즈 바닥을 넘는 행 {int(clearing['rows']):,}"
        )
        qualifier = str(entry.get("qualifier") or "").strip()
        if qualifier:
            line += f"  — 기록된 한정(원문): {qualifier}"
        lines.append(line)
    floor = reader.reversal_floor()
    lines += [
        (
            "  바닥 = head별 median, 풀링 없음 ("
            + ", ".join(
                f"{head} {float(floor['by_head'][head]):.3g}"
                for head in reader.analysed_heads
            )
            + ")"
        ),
        "",
        f"IG–ISM 일치도 (행별 {reader.validation['statistic']}의 median, 판정 없음)",
    ]
    for stratum in reader.strata:
        figures = reader.validation_figures(stratum, primary)
        per_row = figures["per_row"]
        lines.append(
            f"  {stratum:<15} {render.format_value(per_row['median'])}  "
            f"분모 {figures['rows']:,}행 / {figures['distinct_protospacers']:,} 타깃  "
            f"(head={primary})"
        )
    others = [head for head in reader.analysed_heads if head != primary]
    lines += [
        (
            "  층은 합치지 않는다 (pooling 금지). 다른 분석 head는 값이 다르다 — "
            + ", ".join(
                f"{head} "
                + render.format_value(
                    reader.validation_figures(STRATUM, head)["per_row"]["median"]
                )
                for head in others
            )
            + f" ({STRATUM} 층)"
        ),
        "",
        "한계",
        f"  {render.NO_VERDICT}",
        (
            "  rollout은 head 축이 없다 (ADR 0008) — head별 읽기를 줄 수 없고, "
            f"{ROLLOUT_LIMIT}"
        ),
        (
            f"  IG는 {reader.ig_rows.rows:,}행 고정 표본이다 — 사례 선정 전에 자기 "
            f"seed로 고정됐고, 사례 {len(numbered)}개 중 "
            f"{sum(1 for case in numbered.values() if reader.has_ig(int(case['row_index'])))}개만 "
            "그 명단에 든다"
        ),
        (
            "  사례 번호(A·B·C)는 오차·head 불일치로 고른 행이다 — 기술적 해석에만 "
            "쓰고 모집단 요약으로 읽지 않는다"
        ),
        f"  {reader.claim_limit}",
    ]
    return "\n".join(lines) + "\n"
