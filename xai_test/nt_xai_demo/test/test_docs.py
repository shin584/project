"""The demo's written record: ADR 0012, the glossary term, the checklist.

These three documents are what someone reads instead of running the demo, so
they can go stale silently. The checks here are cheap and catch the drift that
would actually mislead: a checklist that tells the meeting to type a command
the shell does not have, a checklist whose `explain` demonstration is no longer
the one case that owns IG, a glossary entry that defines the demo without
warning off the two names it is confused with.

They do not grade the prose, with one exception the ticket asks for: the
checklist must still carry the sentence the presenter says out loud where the
two nucleases are on one screen (#163, "\"이 수치는 비교하지 않는다\"를 말할
지점"). Nothing else here asserts a wording a later edit would have to keep.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from xai_test.nt_xai_demo import shell as sh

#: `xai_test/`, two levels up from this file's package.
XAI_TEST = Path(__file__).resolve().parents[2]

ADR = (
    XAI_TEST
    / "docs"
    / "adr"
    / "0012-nt-xai-demo-presents-stored-artifacts-without-comparing-them.md"
)
CONTEXT = XAI_TEST / "CONTEXT.md"
CHECKLIST = XAI_TEST / "nt_xai_demo" / "demo_checklist.md"

#: The one SpCas9 case inside the fixed 400-row IG list (issue #155, report 3).
IG_CASE = "GEN-HEADGAP-02"


@pytest.fixture(scope="module")
def adr() -> str:
    return ADR.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def checklist() -> str:
    return CHECKLIST.read_text(encoding="utf-8")


def test_adr_0012_records_both_decisions_and_its_relation_to_adr_0003(adr):
    """One ADR, two decisions, and the frame it does not inherit.

    ADR 0003 made the SaCas9 demo CLI comparative on purpose. Reading 0012
    without that relation spelled out, the two look like contradictory house
    styles rather than two demos with different available axes.
    """
    assert "0003-demo-cli-compares-model-a-and-model-b.md" in adr
    # Decision 1: no cross-nuclease numbers, a qualitative PAM finding only.
    assert "qualitative" in adr.lower()
    # Decision 2: stored artifacts only, no model is loaded.
    assert "torch" in adr and "transformers" in adr


def test_context_defines_the_demo_and_warns_off_the_two_confusable_names():
    """`_Avoid_` is the half that does the work here.

    Both rejected names are live in this repo: "the demo" is the SaCas9 Model
    A/B CLI of ADR 0003, and "통합 데모" asserts the comparison this demo
    refuses to make.
    """
    context = CONTEXT.read_text(encoding="utf-8")
    assert "**NT XAI 데모**" in context
    entry = context.split("**NT XAI 데모**", 1)[1].split("\n\n", 1)[0]
    assert "_Avoid_" in entry
    assert "the demo" in entry
    assert "통합 데모" in entry


def test_checklist_demonstrates_explain_on_the_one_case_that_owns_ig(checklist):
    assert IG_CASE in checklist


def test_checklist_marks_where_the_no_comparison_line_is_spoken(checklist):
    assert "비교하지 않는다" in checklist


def test_every_command_the_checklist_tells_the_meeting_to_type_exists(checklist):
    """A checklist command that the shell rejects is a failure mid-meeting.

    The checklist writes each typed command with its REPL prompt
    (`cas12a> report`), which is what makes this check sound: the prompt marks
    which backticked snippets are commands, so a misspelled view name still
    matches as a command and fails here, instead of quietly not being a command
    any more. Commands are read out of the document rather than listed in this
    test, so a step added to the checklist is covered without editing the test.
    """
    prompts = "|".join(sh.NUCLEASES)
    typed = re.findall(rf"`({prompts})> ([^`]+)`", checklist)
    assert typed, "체크리스트에 `<뉴클레아제>> <뷰>` 꼴 명령이 없다"

    seen = set()
    for nuclease, command in typed:
        tokens = command.split()
        view = sh.VIEW_BY_NAME.get(tokens[0])
        assert view is not None, f"없는 뷰: {nuclease}> {command}"
        if view.needs_nuclease:
            assert nuclease in view.builders, f"{nuclease}에 없는 뷰: {command}"
        if view.argument_required:
            assert len(tokens) > 1, f"인자가 빠졌다: {view.usage}"
        seen.add(view.name)

    # Every view the demo offers is walked through at least once: a view
    # nobody is told to open is a view nobody checks before the meeting.
    assert seen == set(sh.VIEW_BY_NAME), (
        f"체크리스트가 다루지 않는 뷰: {sorted(set(sh.VIEW_BY_NAME) - seen)}"
    )
