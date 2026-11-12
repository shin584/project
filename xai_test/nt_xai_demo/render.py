"""정직성 렌더 토대: 표시값마다 층·head·통계 종류·실제 분모를 함께 낸다.

Every view in this demo renders through `Figure`, and a `Figure` cannot be
built without those four fields. The reason is specific to these artifacts:
the same statistic exists in two strata (CONTEXT.md's term; `층` on screen)
over differently-shaped populations - SpCas9's generalisation stratum is 200
rows over 193 targets, the PAM library stratum 200 rows over 5 targets - the
head axis differs between ISM and IG, and ISM and IG are not the same kind of
statistic. A number shown without all four reads as a property of the model
when it is a property of one stratum, one head and one denominator.

The population belongs to the individual figure, not to the analysis: a view
holds figures with different denominators side by side, so the denominator is
printed per line rather than once in a header.

`render` always prints the artifact's `claim_limit` at the bottom and refuses
to render without one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

CLAIM_LIMIT_HEADING = "주장 범위"

#: Each `report` ends with this one line and no other view pointer. Defined
#: here because both nucleases' reports must print the identical pointer.
CROSS_POINTER = "교차 소견은 `cross`"


class RenderRefused(ValueError):
    """A figure or view is missing something that must not be left off."""


def quoted(note: str, gloss: str) -> tuple[str, str]:
    """An artifact's own note, verbatim, with the demo's Korean gloss beneath it.

    These notes are recorded in English while the demo speaks Korean.
    Restating one *in place of* the quote would make the demo the author of a
    limit the analysis stated, so the recorded text is printed verbatim and the
    Korean reading is added beside it, each labelled for what it is. A speaker
    reads the gloss; anyone checking the claim reads the line above it.

    Defined here, like `CROSS_POINTER`, because both nucleases' views have to
    print an artifact's own words in the identical shape.
    """
    return (f"원문(artifact 기록): {note}", f"  └ 우리말 (데모가 붙인 설명): {gloss}")


@dataclass(frozen=True)
class Stratum:
    """Which stratum (층) a figure belongs to, and whether it pools.

    A figure over one stratum names it. A figure over a pooled population has
    to say so *and* say what the pooling does, because that is exactly where
    these numbers mislead: Cas12a's `all` Spearman of 0.7007 carries the mean
    efficiency difference between the TTTV and TTTT groups, while ranking
    within a group is 0.5423 and 0.2995. Declaring the pooling is what makes it
    visible; it is not detectable from the stratum's name.

    SpCas9's two strata are a different case - `ig_ism_validation.json` declares
    `"pooling": "forbidden"` - and nothing may pool them. That is enforced
    where the figures are read, not here: `SpCas9Reader.validation_figures`
    takes one declared stratum at a time and has no combined entry to return.
    """

    name: str
    pooled: bool = False
    pooling_note: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise RenderRefused("층 이름이 비어 있다")
        if self.pooled and not (self.pooling_note or "").strip():
            raise RenderRefused(
                f"층 {self.name!r}은 풀링된 모집단이다: 풀링 효과를 함께 적어야 한다"
            )
        if not self.pooled and self.pooling_note:
            raise RenderRefused(
                f"층 {self.name!r}은 풀링이 아니라고 하면서 풀링 설명을 달고 있다"
            )

    def text(self) -> str:
        return f"{self.name} (풀링: {self.pooling_note})" if self.pooled else self.name


@dataclass(frozen=True)
class Denominator:
    """The count actually divided by, and what those units are.

    `rows` is the number of rows the figure was computed over - not the
    population that was swept, and not the sum of two strata. `targets` is
    recorded where rows and distinct targets differ (the PAM library's 200 rows
    cover 5 targets), because a reader who sees only the row count will read a
    replication factor as a sample size.
    """

    rows: int
    unit: str
    targets: int | None = None
    target_unit: str = "타깃"
    note: str | None = None

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise RenderRefused(f"분모가 {self.rows}행이다: 실제 분모를 적어야 한다")
        if not self.unit.strip():
            raise RenderRefused("분모의 단위 이름이 비어 있다")
        if self.targets is not None and self.targets <= 0:
            raise RenderRefused(f"타깃 수가 {self.targets}이다")

    def text(self) -> str:
        parts = [f"{self.rows:,}{self.unit}"]
        if self.targets is not None:
            parts.append(f"{self.targets:,}{self.target_unit}")
        text = "·".join(parts)
        return f"{text} ({self.note})" if self.note else text


def single_row(note: str = "이 행") -> Denominator:
    """The denominator of a figure read off one row: that one row, said out loud.

    A per-case figure is one row's value, and both nucleases' `explain` views
    need to say so the same way - a figure whose denominator is left implicit
    is the one that gets quoted as a population result.
    """
    return Denominator(rows=1, unit="행", note=note)


@dataclass(frozen=True)
class Figure:
    """One displayed value with the four fields it must never be shown without."""

    name: str
    value: Any
    stratum: Stratum | str
    head: str
    statistic: str
    denominator: Denominator
    scale: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.stratum, str):
            # A bare name is the single-stratum case; pooling has to be stated.
            object.__setattr__(self, "stratum", Stratum(self.stratum))
        for label, text in (
            ("이름", self.name),
            ("head", self.head),
            ("통계 종류", self.statistic),
        ):
            if not str(text).strip():
                raise RenderRefused(f"{self.name or '표시값'}: {label}이(가) 비어 있다")

    def value_text(self) -> str:
        return format_value(self.value)

    def lines(self) -> list[str]:
        stratum = (
            self.stratum if isinstance(self.stratum, Stratum) else Stratum(self.stratum)
        )
        provenance = (
            f"층={stratum.text()} / head={self.head} / 통계={self.statistic} / "
            f"분모={self.denominator.text()}"
        )
        if self.scale:
            provenance += f" / 척도={self.scale}"
        lines = [f"{self.name}: {self.value_text()}", f"  {provenance}"]
        if self.note:
            lines.append(f"  주: {self.note}")
        return lines


@dataclass
class Section:
    """A titled group of figures. Prose goes in `notes`, never in a value.

    `chart` is drawn after the figures: it is the picture of a figure above it
    (a position profile), so the figure carrying its four fields comes first
    and a chart never appears without one.
    """

    title: str
    figures: Sequence[Figure] = field(default_factory=tuple)
    notes: Sequence[str] = field(default_factory=tuple)
    chart: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.chart and not self.figures:
            raise RenderRefused(
                f"[{self.title}] 그림만 있고 그 그림의 표시값이 없다: "
                "층·head·통계·분모를 적을 곳이 없다"
            )

    def lines(self) -> list[str]:
        lines = [f"[{self.title}]"]
        for note in self.notes:
            lines.append(f"  {note}")
        for figure in self.figures:
            lines.extend(f"  {line}" for line in figure.lines())
        lines.extend(f"  {line}" for line in self.chart)
        return lines


#: The PAM bracket drawn beside a profile: opens, continues, closes.
PAM_OPEN, PAM_INSIDE, PAM_CLOSE, PAM_OUTSIDE = "[", "|", "]", " "
BAR = "#"
PEAK_MARK = "<- 최대"


#: The agreement views carry no verdict; both nucleases say so in these words.
NO_VERDICT = (
    "측정된 값으로 제시한다. 합격·불합격 판정을 붙이지 않는다 — 두 기법은 서로의 "
    "정답이 아니다."
)


def profile_peak(values: Sequence[float]) -> int:
    """The position of a profile's largest value, NaN skipped.

    The one definition both the peak figure and the bars' peak mark use, so
    the number a view states and the bar it marks cannot disagree.
    """
    finite = [(float(v), i) for i, v in enumerate(values) if not math.isnan(v)]
    if not finite:
        raise RenderRefused("그릴 값이 하나도 없다")
    return max(finite, key=lambda pair: pair[0])[1]


def profile_bars(
    values: Sequence[float],
    *,
    pam: tuple[int, int],
    width: int = 40,
) -> list[str]:
    """A per-position profile as console ASCII bars, the PAM bracketed.

    One line per position: the PAM bracket, the position, a bar scaled to the
    profile's own peak, and the value. The bracket is what makes the point of
    the picture readable at a glance - whether the most sensitive position
    sits inside the PAM - so it is drawn on every PAM line, not only at its
    ends. The peak is marked once.

    A missing value (NaN) is written as missing, never as an empty bar: an
    empty bar reads as "no effect", which nothing measured.
    """
    start, end = pam
    if not 0 <= start <= end < len(values):
        raise RenderRefused(
            f"PAM 구간 {start}-{end}이 프로파일 {len(values)}개 위치 밖이다"
        )
    peak = profile_peak(values)
    peak_value = float(values[peak])

    lines = [
        (
            f"{PAM_OPEN} {PAM_CLOSE} = PAM (위치 {start}-{end}). "
            "막대 길이는 이 프로파일의 최대값 기준이다"
        )
    ]
    for position, value in enumerate(values):
        if position == start:
            marker = PAM_OPEN
        elif position == end:
            marker = PAM_CLOSE
        elif start < position < end:
            marker = PAM_INSIDE
        else:
            marker = PAM_OUTSIDE
        if math.isnan(value):
            lines.append(f"{marker} {position:>3}  {'':<{width}}  값 없음")
            continue
        length = round(value / peak_value * width) if peak_value > 0 else 0
        if value > 0:
            length = max(length, 1)
        line = f"{marker} {position:>3}  {BAR * length:<{width}}  {value:.4g}"
        if position == peak:
            line += f"  {PEAK_MARK}"
        lines.append(line)
    return lines


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    return str(value)


def render(
    sections: Sequence[Section],
    *,
    claim_limit: str,
    title: str | None = None,
    questions: Sequence[str] = (),
    pointers: Sequence[str] = (),
) -> str:
    """Render sections with the claim limit always at the bottom.

    `questions` is the list of questions the analysis answered, printed at the
    head of a view: the two nucleases' analyses are not symmetric (SpCas9 has a
    noise floor and a head-order reversal, Cas12a has neither), and saying what
    each analysis asked keeps the asymmetry from reading as an omission.
    """
    if not claim_limit.strip():
        raise RenderRefused("claim_limit 없이는 렌더하지 않는다")
    if not sections:
        raise RenderRefused("표시할 섹션이 없다")

    lines: list[str] = []
    if title:
        lines += [title, "=" * max(len(title), 8), ""]
    if questions:
        lines.append("이 분석이 답한 질문")
        lines += [f"  - {question}" for question in questions]
        lines.append("")
    for section in sections:
        lines += section.lines()
        lines.append("")
    for pointer in pointers:
        lines.append(pointer)
    if pointers:
        lines.append("")
    lines.append(f"-- {CLAIM_LIMIT_HEADING} " + "-" * 40)
    lines += [f"  {line}" if line.strip() else "" for line in claim_limit.splitlines()]
    return "\n".join(lines).rstrip() + "\n"
