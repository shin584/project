"""화면 한 장을 그리는 공용 조각 (#169, #171, #173).

The two nucleases' screens print the same shapes - a case number, a sequence
with an ISM bar under it, token rows, one rollout line, one provenance line -
over different artifacts. What differs is which artifact a figure is read
from; what does not differ is how a figure is drawn, and drawing it twice is
how the two screens would quietly stop matching.

So the drawing lives here and the reading stays in each screen module. Nothing
here touches a reader: these take values and return text.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from . import render

__all__ = [
    "LABEL_WIDTH",
    "MISSING",
    "NUMBER_MAX_LENGTH",
    "ROLLOUT_LIMIT",
    "SPARKS",
    "UnknownCaseNumber",
    "bar_definition",
    "bracket_line",
    "inclusive_span",
    "ism_caption",
    "looks_like_case_number",
    "missing_note",
    "number_range",
    "parse_case_number",
    "peak_text",
    "rollout_block",
    "span_text",
    "spark_bar",
]

#: Eight heights for the ISM bar; a missing value is a dot, never a low bar.
SPARKS = "▁▂▃▄▅▆▇█"
MISSING = "·"

#: "서열  " / "ISM   " in terminal columns: what the bar is indented by, so a
#: bracket drawn under a position lands on that position's letter.
LABEL_WIDTH = 6

ROLLOUT_LIMIT = "모델 내부 정보 의존성이며 물리적 결합의 근거가 아님"
#: Rollout has no head axis at all (ADR 0008), so one row of values stands
#: under however many heads a screen shows. Saying so is what stops the
#: numbers being read as the last-named head's.
ROLLOUT_HEAD_INVARIANCE = "head 축이 없어 head별 값이 아니다 (ADR 0008)"

#: Longer than this is read as a sequence even with a digit in it: no case
#: number comes near it, and a pasted window with a typo must hear its unit.
NUMBER_MAX_LENGTH = 4
_NUMBER = re.compile(r"([A-Za-z])(\d+)")


class UnknownCaseNumber(KeyError):
    """A case number the selection does not hold. Its message names the range."""

    def __str__(self) -> str:  # KeyError would quote the message
        return str(self.args[0])


def parse_case_number(typed: str) -> str | None:
    """`a2` -> `A2`, or None when the text is not a letter-and-digits number."""
    match = _NUMBER.fullmatch(typed.strip())
    if match is None:
        return None
    return f"{match.group(1).upper()}{int(match.group(2))}"


def looks_like_case_number(typed: str) -> bool:
    """Route one input: a short input with a digit is a number, else a sequence.

    So a mistyped number still gets the number's range in its refusal, and a
    mistyped sequence - a stray digit included - gets its unit in its own.
    """
    text = typed.strip()
    return len(text) <= NUMBER_MAX_LENGTH and any(c.isdigit() for c in text)


def number_range(numbered: Sequence[str], letters: Sequence[tuple[str, str]]) -> str:
    """The valid numbers by letter: `A1–A3 ERRLO · C2 HEADGAP`.

    A letter whose ranks are contiguous prints as a span; one with a gap
    prints its numbers, because a span would offer numbers that do not exist.
    """
    parts = []
    for letter, role in letters:
        ranks = sorted(int(key[1:]) for key in numbered if key[0] == letter)
        if not ranks:
            continue
        contiguous = ranks == list(range(ranks[0], ranks[-1] + 1))
        span = (
            f"{letter}{ranks[0]}–{letter}{ranks[-1]}"
            if contiguous and len(ranks) > 1
            else ", ".join(f"{letter}{rank}" for rank in ranks)
        )
        parts.append(f"{span} {role}")
    return " · ".join(parts)


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


def span_text(start: int, end: int) -> str:
    """A half-open `[start, end)` span as the inclusive positions a reader counts."""
    return f"{start}" if end - start == 1 else f"{start}-{end - 1}"


def inclusive_span(start: int, end: int) -> tuple[int, int]:
    """An inclusive `[start, end]` boundary as the half-open span used here.

    The two artifacts disagree: Cas12a's `layout.json` records half-open
    `dna_spans`, SpCas9's `ig.sidecar.json` records inclusive
    `token_boundaries`. Converting once, by name, keeps the `+ 1` from being
    written out at each use - which is exactly where an off-by-one would hide.
    """
    return (start, end + 1)


def bracket_line(span: tuple[int, int], label: str) -> str:
    """A bracket under one span of the sequence, with its name to the right."""
    start, end = span
    width = end - start
    bracket = "[" + "-" * max(width - 2, 0) + "]" if width > 1 else "^"
    return " " * (LABEL_WIDTH + start) + bracket + f"  {label}"


def bar_definition(*, delta: str = "Δ") -> str:
    """What the bars are, without any one bar's own numbers.

    `delta` names the stored quantity as its artifact names it (Cas12a's
    `raw_delta`, SpCas9's `delta`), so the caption says which array the bars
    were read from rather than describing them in the demo's own words.
    """
    return f"막대 = 위치별 대체염기 3개 중 최대 |{delta}| (WT 칸 제외)"


def missing_note(profiles: Sequence[Sequence[float]]) -> str:
    """The dot's meaning, added only where a bar actually has one."""
    if any(math.isnan(value) for profile in profiles for value in profile):
        return f" ({MISSING} = 유효 대체염기 없음)"
    return ""


def ism_caption(profile: Sequence[float], *, delta: str = "Δ") -> str:
    """One bar's whole caption: what it is, and this row's own peak."""
    return (
        f"{bar_definition(delta=delta)} · {peak_text(profile)}{missing_note([profile])}"
    )


def peak_text(profile: Sequence[float], label: str = "이 행") -> str:
    """Where this bar's own peak is and how big it is, in the stored units.

    Every bar is scaled to its own maximum - the analyses' own convention,
    because the heads' dynamic ranges differ several-fold and a shared scale
    would report that difference as a difference in *where* a head reads
    (`cases_annotated.json`, `pam_share_definition`). The price is that two
    bars of equal height can stand for very different magnitudes, so a bar is
    never drawn without its own peak figure beside it; `label` names whose
    peak it is when more than one bar is shown.
    """
    peak = render.profile_peak(profile)
    return f"{label} 최대 {profile[peak]:.3f} @ 위치 {peak}"


def rollout_block(
    referenced: Sequence[float],
    *,
    queries: int,
    query_name: str = "DNA",
    head_note: str = "",
) -> list[str]:
    """The rollout table: one line of per-token values, then its one limit line.

    `queries` is how many query rows were averaged, which is the matrix's row
    count - not the number of values returned, which is its column count. The
    two coincide on a square block and would silently diverge on anything else.
    """
    return [
        f"Attention Rollout (토큰별 참조되는 정도, {query_name} query {queries}개 평균"
        + (f", {head_note}" if head_note else "")
        + ")",
        "  "
        + "  ".join(
            f"t{index} {value:.3f}" for index, value in enumerate(referenced, 1)
        ),
        f"  {ROLLOUT_LIMIT}",
    ]
