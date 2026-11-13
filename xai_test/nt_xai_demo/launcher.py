"""데모 런처: 뉴클레아제를 첫 인자로 받고, 그 한 분석만 띄운다 (#169, #171).

`python -m xai_test.nt_xai_demo cas12a` opens the `cas12a>` REPL;
`python -m xai_test.nt_xai_demo cas12a A1` prints the same screen once. Both go
through `Session.execute`, so the one-shot output and what the meeting types
into the REPL cannot diverge.

The nuclease is fixed when the demo starts. There is no switching inside a
session: two nucleases never share a screen, which is ADR 0012's
no-comparison decision made structural. SpCas9 says it is not ready until its
own screen lands (#173).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import cas12a_screen, render
from .reader import DEFAULT_DATA_DIR, ArtifactMismatch, Cas12aReader

QUIT_WORDS = ("quit", "exit", "q", "종료")

#: What each nuclease is, as the launcher's usage lists it.
NUCLEASES: dict[str, str] = {
    "cas12a": "NT-Cas12a (Test 1,500행 본 분석)",
    "spcas9": "9-Hydra v3 (SpCas9)",
}
#: Nucleases whose screen is not built yet, with the ticket that builds it.
NOT_READY: dict[str, str] = {"spcas9": "#173"}


@dataclass(frozen=True)
class Outcome:
    """What one input produced. `ok` False makes a one-shot call exit non-zero."""

    text: str = ""
    stop: bool = False
    ok: bool = True


@dataclass
class Session:
    """One Cas12a session: the reader, built once, and the inputs it takes."""

    data_dir: Path = DEFAULT_DATA_DIR
    _reader: Cas12aReader | None = field(default=None, repr=False)

    prompt = "cas12a> "

    @property
    def reader(self) -> Cas12aReader:
        if self._reader is None:
            self._reader = Cas12aReader(self.data_dir)
        return self._reader

    def help_text(self) -> str:
        return "\n".join(
            [
                f"{NUCLEASES['cas12a']} — 저장된 산출물만 읽는다 (모델 로드 없음)",
                f"  <번호>   사례 한 장: {cas12a_screen.number_range(self.reader)}",
                "  report   성능 + IG–ISM 일치도 + 한계",
                "  help     이 안내",
                "  quit     종료",
            ]
        )

    def execute(self, line: str) -> Outcome:
        """Run one input line. The only path a screen's text comes out of."""
        parts = line.strip().split()
        if not parts:
            return Outcome()
        command = parts[0].lower()
        if command in QUIT_WORDS:
            return Outcome(stop=True)
        if len(parts) > 1:
            return Outcome(
                f"입력은 하나씩 받는다: 번호 하나, report, help, quit. 받은 것 `{line.strip()}`",
                ok=False,
            )
        try:
            if command == "help":
                return Outcome(self.help_text())
            if command == "report":
                return Outcome(cas12a_screen.report(self.reader))
            return Outcome(cas12a_screen.case_screen(self.reader, parts[0]))
        except cas12a_screen.UnknownCaseNumber as error:
            return Outcome(str(error), ok=False)
        except (
            render.RenderRefused,
            ArtifactMismatch,
            FileNotFoundError,
            KeyError,
        ) as error:
            return Outcome(f"화면을 낼 수 없다: {error}", ok=False)


def usage() -> str:
    lines = ["사용법: python -m xai_test.nt_xai_demo <뉴클레아제> [번호|report]", ""]
    for name, label in NUCLEASES.items():
        ready = f"  (준비 안 됨, {NOT_READY[name]})" if name in NOT_READY else ""
        lines.append(f"  {name:<8} {label}{ready}")
    lines += [
        "",
        "  prepare  데모 자료를 data/에 모으고 manifest를 만든다",
        "  verify   data/를 manifest와 대조한다",
        "",
        "예: cas12a (REPL), cas12a A1 (사례 한 장), cas12a report",
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
    if nuclease in NOT_READY:
        error(
            f"{nuclease}: 준비 안 됨 — 사례 화면은 {NOT_READY[nuclease]}에서 만든다. "
            "지금은 cas12a만 띄운다."
        )
        return 1
    session = Session(data_dir=data_dir)
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
        write(f"Cas12a 자료를 읽을 수 없다: {error}")
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
