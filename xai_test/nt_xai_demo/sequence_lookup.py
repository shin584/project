"""Raw sequence -> stored row, for whichever nuclease's demo (#169, #172).

The demo's second input is the sequence itself. This module knows only row
IDs and their sequences: the unit (length, name on screen) and the population
are the caller's, so Cas12a's 34bp target window and SpCas9's 30bp window
(#174) share the same lookup, duplicate handling and miss message.

The same sequence can sit on several rows - cell-line replicates measured
separately. The screen shows the smallest row ID and says how many rows share
the target; it never averages them or picks by anything else.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

ALPHABET = "ACGT"
#: How many stored sequences a miss offers to copy from.
EXAMPLES = 3


class NotASequence(ValueError):
    """The input is not in the demo's input unit. The message names the unit."""


class SequenceNotStored(KeyError):
    """A well-formed sequence that no stored row holds."""

    def __str__(self) -> str:  # KeyError would quote the message
        return str(self.args[0])


@dataclass(frozen=True)
class SequenceUnit:
    """What a typed sequence must be: its length and how the screen names it."""

    length: int
    name: str
    #: (length, sentence) pairs: a wrong length the caller recognises - another
    #: unit of its own a user is likely to paste - gets that sentence appended.
    hints: tuple[tuple[int, str], ...] = ()

    def check(self, typed: str) -> str:
        """The sequence normalised to upper case, or a refusal naming the unit."""
        sequence = typed.strip().upper()
        if len(sequence) != self.length:
            refusal = (
                f"{self.name}이 아니다: 길이 {len(sequence)} ({self.length}이어야 한다)"
            )
            hint = dict(self.hints).get(len(sequence))
            if hint:
                refusal += f" — {hint}"
            raise NotASequence(refusal)
        for position, letter in enumerate(sequence):
            if letter not in ALPHABET:
                raise NotASequence(
                    f"{self.name}이 아니다: {ALPHABET} 외 문자 {letter!r} (위치 {position})"
                )
        return sequence


@dataclass(frozen=True)
class SequenceHit:
    """Every stored row holding one sequence, smallest row ID first."""

    sequence: str
    rows: tuple[int, ...]
    id_name: str = "row"

    @property
    def row(self) -> int:
        """The row the screen shows: the smallest ID."""
        return self.rows[0]

    def duplicate_line(self) -> str | None:
        """One line when the target is on more than one row; None otherwise."""
        if len(self.rows) == 1:
            return None
        listed = ", ".join(str(row) for row in self.rows)
        return (
            f"이 타깃은 {len(self.rows)}행 중복(세포주 반복): {self.id_name} {listed} — "
            f"가장 작은 {self.id_name} {self.row}를 보여준다"
        )


class SequenceIndex:
    """The stored population's sequences, looked up by the typed text."""

    def __init__(
        self,
        sequences: Mapping[int, str],
        unit: SequenceUnit,
        *,
        id_name: str = "row",
        scope: str | None = None,
    ) -> None:
        if not sequences:
            raise ValueError("조회할 저장 서열이 하나도 없다")
        self.unit = unit
        self.id_name = id_name
        #: What the stored rows are, when they are not every row the nuclease
        #: has - named in a miss, so "not stored" is not read as "not measured".
        self.scope = scope
        self.stored_rows = len(sequences)
        by_sequence: dict[str, list[int]] = {}
        for row in sorted(int(row) for row in sequences):
            try:
                sequence = unit.check(sequences[row])
            except NotASequence as error:
                raise ValueError(f"저장된 {id_name} {row}: {error}") from None
            by_sequence.setdefault(sequence, []).append(row)
        # Built in row order, so the first key is the smallest row's sequence.
        self._rows = {seq: tuple(rows) for seq, rows in by_sequence.items()}

    @property
    def distinct(self) -> int:
        return len(self._rows)

    def examples(self, count: int = EXAMPLES) -> tuple[str, ...]:
        """Stored sequences to copy, those of the smallest row IDs first."""
        return tuple(list(self._rows)[:count])

    def find(self, typed: str) -> SequenceHit:
        sequence = self.unit.check(typed)
        rows = self._rows.get(sequence)
        if rows is None:
            counted = f"저장된 {self.stored_rows:,}행에 없음"
            if self.distinct != self.stored_rows:
                counted += f" (서로 다른 서열 {self.distinct:,}개)"
            if self.scope:
                counted += f" — 조회 범위: {self.scope}"
            raise SequenceNotStored(
                f"{counted}: {sequence}. 예: {', '.join(self.examples())}"
            )
        return SequenceHit(sequence, rows, self.id_name)
