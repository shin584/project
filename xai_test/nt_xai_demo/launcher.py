"""데모 런처: 뉴클레아제를 첫 인자로 받고, 그 한 분석만 띄운다 (#169, #171, #173).

`python -m xai_test.nt_xai_demo cas12a` opens the `cas12a>` REPL;
`python -m xai_test.nt_xai_demo cas12a A1` prints the same screen once. Both go
through `Session.execute`, so the one-shot output and what the meeting types
into the REPL cannot diverge.

The nuclease is fixed when the demo starts. There is no switching inside a
session: two nucleases never share a screen, which is ADR 0012's
no-comparison decision made structural rather than stated in a caption.

What differs between the two is held in a `Nuclease` record - its reader, its
screen module, and the input its prompt names - so the loop, the refusals and
the exit codes are written once and both nucleases get the same ones.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from . import cas12a_screen, render, spcas9_screen
from .reader import DEFAULT_DATA_DIR, ArtifactMismatch, Cas12aReader, SpCas9Reader
from .screen_parts import UnknownCaseNumber
from .sequence_lookup import NotASequence, SequenceNotStored

QUIT_WORDS = ("quit", "exit", "q", "종료")


@dataclass(frozen=True)
class Nuclease:
    """One analysis the demo can open: what to read it with, and what to call it.

    `screen` is the module that turns an input into the screen's text. Both
    screen modules present the same four names - `screen_for`, `report`,
    `help_lines` and `number_range` - so this launcher never asks which
    nuclease it is holding. The refusals it catches are imported from the
    modules that define them rather than read off `screen`, so a screen that
    cannot raise one does not have to re-export it to be launchable.
    """

    key: str
    label: str
    #: What the prompt names as acceptable input, in this nuclease's own unit.
    takes: str
    reader_factory: Callable[[Path], Any]
    screen: ModuleType

    @property
    def prompt(self) -> str:
        # The prompt names the unit: another nuclease's sequence, of another
        # length, pasted here must not look like it should have matched.
        return f"{self.key} [{self.takes}]> "


NUCLEASES: dict[str, Nuclease] = {
    "cas12a": Nuclease(
        key="cas12a",
        label="NT-Cas12a (Test 1,500행 본 분석)",
        takes="번호 | 34bp 타깃 서열",
        reader_factory=Cas12aReader,
        screen=cas12a_screen,
    ),
    "spcas9": Nuclease(
        key="spcas9",
        label="9-Hydra v3 (SpCas9, 분석 head 3개)",
        # The raw 30bp input lands in #174; until then the prompt offers the
        # numbers it can actually open rather than an input that would refuse.
        takes="번호",
        reader_factory=SpCas9Reader,
        screen=spcas9_screen,
    ),
}


@dataclass(frozen=True)
class Outcome:
    """What one input produced. `ok` False makes a one-shot call exit non-zero."""

    text: str = ""
    stop: bool = False
    ok: bool = True


@dataclass
class Session:
    """One nuclease's session: the reader, built once, and the inputs it takes."""

    nuclease: Nuclease
    data_dir: Path = DEFAULT_DATA_DIR
    _reader: Any | None = field(default=None, repr=False)

    @property
    def prompt(self) -> str:
        return self.nuclease.prompt

    @property
    def reader(self) -> Any:
        if self._reader is None:
            self._reader = self.nuclease.reader_factory(self.data_dir)
        return self._reader

    def help_text(self) -> str:
        screen = self.nuclease.screen
        return "\n".join(
            [
                f"{self.nuclease.label} — 저장된 산출물만 읽는다 (모델 로드 없음)",
                *screen.help_lines(self.reader),
                "  report   성능 + IG–ISM 일치도 + 한계",
                "  help     이 안내",
                "  quit     종료",
            ]
        )

    def execute(self, line: str) -> Outcome:
        """Run one input line. The only path a screen's text comes out of."""
        screen = self.nuclease.screen
        parts = line.strip().split()
        if not parts:
            return Outcome()
        command = parts[0].lower()
        if command in QUIT_WORDS:
            return Outcome(stop=True)
        if len(parts) > 1:
            return Outcome(
                "입력은 하나씩 받는다: 번호 하나, 서열 하나, report, help, quit. "
                f"받은 것 `{line.strip()}`",
                ok=False,
            )
        try:
            if command == "help":
                return Outcome(self.help_text())
            if command == "report":
                return Outcome(screen.report(self.reader))
            return Outcome(screen.screen_for(self.reader, parts[0]))
        except (UnknownCaseNumber, SequenceNotStored) as error:
            return Outcome(str(error), ok=False)
        except NotASequence as error:
            # Letters only, so it was read as a sequence; it may have been a
            # number typed wrong, so the refusal names both inputs.
            return Outcome(
                f"{error}. 사례 번호라면: {screen.number_range(self.reader)}",
                ok=False,
            )
        except (
            render.RenderRefused,
            ArtifactMismatch,
            FileNotFoundError,
            KeyError,
        ) as error:
            return Outcome(f"화면을 낼 수 없다: {error}", ok=False)


def usage() -> str:
    lines = [
        "사용법: python -m xai_test.nt_xai_demo <뉴클레아제> [번호|서열|report]",
        "",
    ]
    for nuclease in NUCLEASES.values():
        lines.append(f"  {nuclease.key:<8} {nuclease.label}  ({nuclease.takes})")
    lines += [
        "",
        "  prepare  데모 자료를 data/에 모으고 manifest를 만든다",
        "  verify   data/를 manifest와 대조한다",
        "",
        (
            "예: cas12a (REPL), cas12a A1 (사례 한 장), cas12a <34bp 타깃 서열>, "
            "spcas9 B1, spcas9 report"
        ),
    ]
    return "\n".join(lines)


def run(
    nuclease: str,
    arguments: Sequence[str] = (),
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    write_error: Callable[[str], None] | None = None,
) -> int:
    """One nuclease's demo: the REPL with no arguments, else one input once."""
    error = write_error or write
    if nuclease not in NUCLEASES:
        error(f"모르는 뉴클레아제다: {nuclease!r}\n\n{usage()}")
        return 2
    session = Session(nuclease=NUCLEASES[nuclease], data_dir=data_dir)
    if arguments:
        outcome = session.execute(" ".join(arguments))
        if outcome.text:
            (write if outcome.ok else error)(outcome.text.rstrip("\n"))
        return 0 if outcome.ok else 1
    return repl(session, read=read, write=write)


def repl(
    session: Session,
    *,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    """The interactive loop. `read`/`write` are injected so tests can drive it."""
    try:
        write(session.help_text())
    except (FileNotFoundError, ArtifactMismatch, KeyError) as error:
        write(f"{session.nuclease.label} 자료를 읽을 수 없다: {error}")
        return 1
    while True:
        try:
            line = read(session.prompt)
        except (EOFError, KeyboardInterrupt):
            write("")
            return 0
        outcome = session.execute(line)
        if outcome.text:
            write(outcome.text.rstrip("\n"))
        if outcome.stop:
            return 0
