"""데모 셸: REPL이 주, 일회성 서브커맨드가 병행.

The demo holds two analyses that are deliberately not compared, so it opens
with no nuclease selected (`(선택 안 됨)>`): choosing one is a visible step and
the prompt afterwards (`cas12a>` / `spcas9>`) says which analysis is on screen.

`run_once` and the REPL both go through `Shell.execute`, so a one-shot
subcommand prints exactly what the REPL prints - the meeting and a shell
transcript cannot diverge.

Views land ticket by ticket. A view that is not built yet says so and names
its ticket rather than printing an empty frame.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import cas12a_views, render
from .reader import DEFAULT_DATA_DIR, Cas12aReader

UNSELECTED_PROMPT = "(선택 안 됨)"

NUCLEASES: dict[str, str] = {
    "cas12a": "NT-Cas12a (Test 1,500행 본 분석)",
    "spcas9": "9-Hydra v3 (SpCas9, 분석된 head 3개)",
}


@dataclass(frozen=True)
class View:
    """One view, with everything the shell needs to offer and run it.

    Keeping the argument, the nuclease requirement and the per-nuclease
    builders in one record means adding a view is one entry here, not an edit
    in four parallel tables.
    """

    name: str
    what: str
    argument: str | None = None
    needs_nuclease: bool = True
    # Per nuclease: the builder that renders it, or a ticket if it is not built
    # yet. A ticket is printed as a ticket, so nobody mistakes a pending view
    # for an analysis that found nothing.
    builders: dict[str, Callable[..., str] | str] = field(default_factory=dict)

    @property
    def usage(self) -> str:
        return f"{self.name} {self.argument}" if self.argument else self.name


VIEWS: tuple[View, ...] = (
    View(
        "report",
        "그 뉴클레아제의 분석 요지",
        builders={"cas12a": cas12a_views.report, "spcas9": "#159"},
    ),
    View(
        "cases",
        "기술 사례 목록 (원 자료 ID 그대로)",
        builders={"cas12a": "#160", "spcas9": "#160"},
    ),
    View(
        "explain",
        "한 행의 방법별 유무와 값",
        argument="<ID>",
        builders={"cas12a": "#160", "spcas9": "#160"},
    ),
    View(
        "agreement",
        "IG-ISM 일치도와 위치 프로파일",
        builders={"cas12a": "#161", "spcas9": "#161"},
    ),
    View(
        "cross",
        "두 모델의 교차 소견 (뉴클레아제 선택과 무관)",
        needs_nuclease=False,
        builders={"cas12a": "#162", "spcas9": "#162", "": "#162"},
    ),
)

VIEW_BY_NAME: dict[str, View] = {view.name: view for view in VIEWS}

QUIT_WORDS = ("quit", "exit", "q", "종료")


@dataclass(frozen=True)
class Outcome:
    """What one command produced: text, whether to stop, whether it worked.

    `ok` is False when the command could not deliver what was asked - an
    unknown command, a view that needs a nuclease, a view that is not built
    yet, a reader that refused. The REPL keeps going either way; a one-shot
    call exits non-zero, like `prepare` and `verify` do.
    """

    text: str = ""
    stop: bool = False
    ok: bool = True


@dataclass
class Shell:
    """The demo shell's state: which data directory, and which nuclease."""

    data_dir: Path = DEFAULT_DATA_DIR
    selected: str | None = None
    _readers: dict[str, Cas12aReader] = field(default_factory=dict, repr=False)

    # -- prompt and help ---------------------------------------------------- #

    @property
    def prompt(self) -> str:
        return f"{self.selected or UNSELECTED_PROMPT}> "

    def help_text(self) -> str:
        lines = ["뉴클레아제 선택"]
        lines += [f"  {name:<8} {label}" for name, label in NUCLEASES.items()]
        lines += ["", "뷰"]
        lines += [f"  {view.usage:<14} {view.what}" for view in VIEWS]
        lines += ["", "그 밖에", "  help            이 안내", "  quit / exit     종료"]
        selected = self.selected or f"{UNSELECTED_PROMPT} (뷰를 보려면 먼저 선택한다)"
        lines += ["", f"현재 선택: {selected}"]
        return "\n".join(lines)

    def greeting(self) -> str:
        return (
            "NT XAI 데모 — NT fine-tune 두 개의 저장된 XAI 산출물을 읽어서 보여준다.\n"
            "모델을 로드하지 않고 두 뉴클레아제의 수치를 비교하지도 않는다.\n\n"
            f"{self.help_text()}"
        )

    # -- readers ------------------------------------------------------------ #

    def reader(self, nuclease: str) -> Cas12aReader:
        """The reader for one nuclease, built once per shell.

        Only Cas12a has a built view in this ticket; the SpCas9 reader joins
        when its views do (#159), so asking for it now is a programming error,
        not a user-facing one.
        """
        if nuclease != "cas12a":
            raise KeyError(f"리더가 아직 없다: {nuclease}")
        if nuclease not in self._readers:
            self._readers[nuclease] = Cas12aReader(self.data_dir)
        return self._readers[nuclease]

    # -- dispatch ----------------------------------------------------------- #

    def execute(self, line: str) -> Outcome:
        """Run one command line. The only path a view's text comes out of."""
        parts = line.strip().split()
        if not parts:
            return Outcome()
        command, arguments = parts[0].lower(), parts[1:]

        if command in QUIT_WORDS:
            return Outcome(stop=True)
        if command == "help":
            return Outcome(self.help_text())
        if command in NUCLEASES:
            self.selected = command
            return Outcome(
                f"{NUCLEASES[command]} 선택. 뷰: "
                + ", ".join(view.usage for view in VIEWS)
            )
        if command in VIEW_BY_NAME:
            return self._view(VIEW_BY_NAME[command], arguments)
        return Outcome(
            f"모르는 명령이다: {parts[0]!r}. `help`로 명령 목록을 본다.", ok=False
        )

    def _view(self, view: View, arguments: Sequence[str]) -> Outcome:
        if view.needs_nuclease and self.selected is None:
            return Outcome(
                "뉴클레아제를 먼저 선택한다: cas12a 또는 spcas9. "
                f"({view.name} 뷰는 한 분석 안에서만 뜻이 있다.)",
                ok=False,
            )
        builder = view.builders.get(self.selected or "")
        if builder is None:
            return Outcome(f"{view.name} 뷰는 {self.selected}에 없다.", ok=False)
        if isinstance(builder, str):  # a ticket, not a builder
            return Outcome(
                f"{view.name} 뷰는 아직 만들지 않았다 ({builder}).", ok=False
            )
        if view.argument and not arguments:
            return Outcome(f"{view.name}은 인자를 받는다: {view.usage}", ok=False)
        try:
            return Outcome(builder(self.reader(self.selected or ""), *arguments))
        except (render.RenderRefused, FileNotFoundError, KeyError) as error:
            return Outcome(f"{view.name} 뷰를 낼 수 없다: {error}", ok=False)


def run_once(argv: Sequence[str], *, data_dir: Path = DEFAULT_DATA_DIR) -> Outcome:
    """One command through the same `execute` the REPL uses.

    `cas12a report` selects and then runs, so a one-shot call and a REPL
    session print the same text for the same view.
    """
    shell = Shell(data_dir=data_dir)
    tokens = list(argv)
    if tokens and tokens[0] in NUCLEASES:
        shell.execute(tokens.pop(0))
    if not tokens:
        return Outcome(shell.help_text())
    return shell.execute(" ".join(tokens))


def repl(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    """The interactive loop. `read`/`write` are injected so tests can drive it."""
    shell = Shell(data_dir=data_dir)
    write(shell.greeting())
    while True:
        try:
            line = read(shell.prompt)
        except (EOFError, KeyboardInterrupt):
            write("")
            return 0
        outcome = shell.execute(line)
        if outcome.text:
            write(outcome.text)
        if outcome.stop:
            return 0
