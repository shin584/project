"""The REPL and the one-shot subcommand are the same views, one code path.

The shell starts with no nuclease selected, because the demo holds two
analyses that are not comparable: picking one is a deliberate step, and the
prompt says which one is being read. Everything a view prints goes through
`Shell.execute`, so a one-shot subcommand cannot drift from what the meeting
sees in the REPL.
"""

from __future__ import annotations

import pytest

from xai_test.nt_xai_demo import shell as sh
from xai_test.nt_xai_demo.test import reader_fixtures as fx


@pytest.fixture
def data_dir(tmp_path):
    return fx.build_data_dir(tmp_path / "data")


@pytest.fixture
def repl(data_dir):
    return sh.Shell(data_dir=data_dir)


def test_it_starts_with_no_nuclease_selected(repl):
    assert repl.selected is None
    assert repl.prompt == "(선택 안 됨)> "


def test_the_greeting_prints_help_and_how_to_select(repl):
    greeting = repl.greeting()

    assert "cas12a" in greeting and "spcas9" in greeting
    assert "report" in greeting
    for view in ("cases", "explain", "agreement", "cross"):
        assert view in greeting


def test_selecting_a_nuclease_changes_the_prompt(repl):
    outcome = repl.execute("cas12a")

    assert repl.selected == "cas12a"
    assert repl.prompt == "cas12a> "
    assert "NT-Cas12a" in outcome.text


def test_the_other_nuclease_is_selectable_too(repl):
    repl.execute("cas12a")
    repl.execute("spcas9")

    assert repl.selected == "spcas9"
    assert repl.prompt == "spcas9> "


def test_a_view_before_selecting_says_to_select_first(repl):
    outcome = repl.execute("report")

    assert "선택" in outcome.text
    assert "cas12a" in outcome.text and "spcas9" in outcome.text
    assert repl.selected is None


def test_an_unknown_command_is_named_and_the_shell_stays_open(repl):
    outcome = repl.execute("predict")

    assert "predict" in outcome.text
    assert not outcome.stop


def test_an_empty_line_does_nothing(repl):
    outcome = repl.execute("   ")

    assert outcome.text == ""
    assert not outcome.stop


def test_quit_stops_the_loop(repl):
    assert repl.execute("quit").stop
    assert repl.execute("exit").stop


def test_help_lists_the_five_views_and_the_current_selection(repl):
    repl.execute("cas12a")
    text = repl.execute("help").text

    for view in ("report", "cases", "explain", "agreement", "cross"):
        assert view in text
    assert "cas12a" in text


def test_the_cas12a_report_runs_from_the_repl(repl):
    repl.execute("cas12a")
    text = repl.execute("report").text

    assert "이 분석이 답한 질문" in text
    assert "0.7007" in text


def test_the_same_view_runs_as_a_one_shot_subcommand(repl, data_dir):
    repl.execute("cas12a")
    from_repl = repl.execute("report").text

    one_shot = sh.run_once(["cas12a", "report"], data_dir=data_dir)

    assert one_shot.text == from_repl


def test_a_one_shot_view_without_a_nuclease_says_to_select_one(data_dir):
    outcome = sh.run_once(["report"], data_dir=data_dir)

    assert "선택" in outcome.text


def test_a_view_this_ticket_has_not_built_yet_names_its_ticket(repl):
    repl.execute("cas12a")

    for view, ticket in (("agreement", "#161"), ("cross", "#162")):
        text = repl.execute(view).text
        assert ticket in text, view


def test_both_nucleases_have_report_cases_and_explain(repl):
    """`report`, `cases` and `explain` are the views both nucleases now carry."""
    for nuclease, case_id in (("cas12a", "4:out_of_range"), ("spcas9", "GEN-ERRHI-01")):
        repl.execute(nuclease)

        for command in ("report", "cases", f"explain {case_id}"):
            outcome = repl.execute(command)

            assert outcome.ok, (nuclease, command)
            assert "이 분석이 답한 질문" in outcome.text, (nuclease, command)


def test_explain_without_an_id_says_what_it_takes(repl):
    repl.execute("spcas9")

    outcome = repl.execute("explain")

    assert not outcome.ok
    assert "<ID>" in outcome.text


def test_explain_with_an_unknown_id_names_the_ids_that_exist(repl):
    repl.execute("spcas9")

    outcome = repl.execute("explain GEN-NOPE-01")

    assert not outcome.ok
    assert "GEN-ERRHI-01" in outcome.text


def test_a_command_that_could_not_deliver_is_marked_not_ok(repl):
    """`prepare` and `verify` exit non-zero on failure; a view has to too."""
    assert not repl.execute("predict").ok  # unknown command
    assert not repl.execute("report").ok  # no nuclease selected
    repl.execute("cas12a")
    assert not repl.execute("agreement").ok  # pending view
    assert repl.execute("report").ok
    assert repl.execute("help").ok


def test_one_shot_help_is_the_shells_help_not_argparses(data_dir):
    text = sh.run_once(["help"], data_dir=data_dir).text

    assert "cas12a" in text and "cross" in text


def test_a_one_shot_call_naming_only_a_nuclease_prints_its_views(data_dir):
    text = sh.run_once(["spcas9"], data_dir=data_dir).text

    assert "report" in text


def test_cross_does_not_ask_for_a_nuclease(repl):
    """`cross` is the one view that stands outside a single analysis."""
    text = repl.execute("cross").text

    assert "먼저 선택" not in text
    assert "#162" in text


def test_the_loop_greets_once_and_stops_on_quit(data_dir):
    typed = iter(["cas12a", "report", "quit", "report"])
    printed: list[str] = []

    status = sh.repl(
        data_dir=data_dir,
        read=lambda prompt: next(typed),
        write=printed.append,
    )

    assert status == 0
    assert "(선택 안 됨)" in printed[0]  # the greeting, before any selection
    assert sum("이 분석이 답한 질문" in text for text in printed) == 1


def test_the_loop_ends_quietly_on_end_of_input(data_dir):
    def read(prompt: str) -> str:
        raise EOFError

    assert sh.repl(data_dir=data_dir, read=read, write=lambda text: None) == 0


def test_a_pending_view_says_pending_rather_than_printing_an_empty_frame(repl):
    repl.execute("cas12a")
    text = repl.execute("agreement").text

    assert "아직" in text
    assert "주장 범위" not in text  # no claim-limit frame around nothing
