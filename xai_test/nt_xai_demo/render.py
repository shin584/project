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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

CLAIM_LIMIT_HEADING = "주장 범위"


class RenderRefused(ValueError):
    """A figure or view is missing something that must not be left off."""


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
    """A titled group of figures. Prose goes in `notes`, never in a value."""

    title: str
    figures: Sequence[Figure] = field(default_factory=tuple)
    notes: Sequence[str] = field(default_factory=tuple)

    def lines(self) -> list[str]:
        lines = [f"[{self.title}]"]
        for note in self.notes:
            lines.append(f"  {note}")
        for figure in self.figures:
            lines.extend(f"  {line}" for line in figure.lines())
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
