"""The demo's written record: ADR 0012, the glossary term, the checklist.

These three documents are what someone reads instead of running the demo, so
they can go stale silently. The checks here are cheap and catch the drift that
would actually mislead: a checklist that tells the meeting to type something
the demo refuses, a checklist whose demonstration case is no longer the one
that owns IG, a glossary entry that defines the demo without warning off the
two names it is confused with.

Since #175 the demo takes two inputs and no view names, so what a checklist can
get wrong changed shape: not a view that was removed, but an input in the wrong
unit - a 20bp protospacer where the screen wants the 30bp window, or a retired
view name that now reads as a sequence. Both fail here.

They do not grade the prose, with one exception the ticket asks for: the
checklist must still carry the sentence the presenter says out loud when asked
which model is better (#163, "\"이 수치는 비교하지 않는다\"를 말할 지점").
Nothing else here asserts a wording a later edit would have to keep.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from xai_test.nt_xai_demo import launcher as ln
from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo import screen_parts

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

#: Each nuclease's typed-sequence unit, which the launcher's record does not
#: carry. Asserted to cover `NUCLEASES`, so a third nuclease fails here rather
#: than having the sequences in its steps silently unchecked.
SEQUENCE_UNITS = {
    "cas12a": rd.CAS12A_SEQUENCE_UNIT,
    "spcas9": rd.SPCAS9_SEQUENCE_UNIT,
}

#: Inputs that are neither a case number nor a sequence, as `Session.execute`
#: branches on them.
WORDS = ("report", "help", *ln.QUIT_WORDS)

#: `` `cas12a> B1` `` - a typed input with the prompt that says where to type it.
TYPED = re.compile(rf"`({'|'.join(ln.NUCLEASES)})> ([^`]+)`")


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


def test_checklist_demonstrates_the_one_case_that_owns_ig(checklist):
    assert IG_CASE in checklist


def test_checklist_marks_where_the_no_comparison_line_is_spoken(checklist):
    assert "비교하지 않는다" in checklist


def test_sequence_units_cover_every_nuclease_the_launcher_offers():
    assert set(SEQUENCE_UNITS) == set(ln.NUCLEASES)


def test_every_input_the_checklist_tells_the_meeting_to_type_is_one_the_demo_takes(
    checklist,
):
    """An input the demo refuses is a failure mid-meeting.

    The checklist writes each typed input with its nuclease's prompt
    (`cas12a> B1`), which is what makes this check sound: the prompt marks
    which backticked snippets are inputs. Each then goes through the two
    decisions `Session.execute` and `screen_for` make - one token only, then
    number-or-sequence - so a retired view name (`cases`, `agreement`) now
    reads as a sequence and fails on its length, and a 20bp protospacer
    written where the 30bp window belongs fails the same way.

    What this cannot check without the bulk data is whether a case number is
    in the selection; `test_reader_real_data.py` covers that against the real
    cases.
    """
    typed = TYPED.findall(checklist)
    assert typed, "체크리스트에 `<뉴클레아제>> <입력>` 꼴 입력이 없다"

    numbers: set[tuple[str, str]] = set()
    sequences: set[tuple[str, str]] = set()
    for nuclease, raw in typed:
        tokens = raw.split()
        assert len(tokens) == 1, f"입력은 하나씩 받는다: {nuclease}> {raw}"
        text = tokens[0]
        if text.lower() in WORDS:
            continue
        if screen_parts.looks_like_case_number(text):
            assert screen_parts.parse_case_number(text) is not None, text
            numbers.add((nuclease, text))
            continue
        # Read as a sequence, so it has to be one in that nuclease's unit:
        # `check` raises `NotASequence`, with the unit named, when it is not.
        SEQUENCE_UNITS[nuclease].check(text)
        sequences.add((nuclease, text))

    # Both inputs are walked through at least once: an input nobody is told to
    # type is an input nobody checks before the meeting.
    assert numbers, "번호로 화면을 여는 단계가 없다"
    assert sequences, "생서열로 화면을 여는 단계가 없다"


def test_the_checklist_opens_a_screen_for_both_nucleases(checklist):
    """Neither nuclease is left out, whichever input opens its screen."""
    opened = {
        nuclease
        for nuclease, raw in TYPED.findall(checklist)
        if raw.split()[0].lower() not in WORDS
    }
    missing = sorted(set(ln.NUCLEASES) - opened)
    assert not missing, f"화면을 열지 않는 뉴클레아제: {missing}"
