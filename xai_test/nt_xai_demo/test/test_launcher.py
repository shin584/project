"""The new launcher: the nuclease is the first argument and fixes the session.

`cas12a A1` and typing `A1` at the `cas12a>` prompt go through the same
`Session.execute`, so what the meeting sees cannot drift from a one-shot run.
"""

from __future__ import annotations

import pytest

from xai_test.nt_xai_demo import __main__ as entry
from xai_test.nt_xai_demo import cas12a_screen, spcas9_screen
from xai_test.nt_xai_demo import launcher as ln
from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo.test import reader_fixtures as fx


@pytest.fixture
def data_dir(tmp_path):
    return fx.build_data_dir(tmp_path / "data")


def _repl(data_dir, inputs):
    """Drive the REPL with `inputs`, returning everything it wrote."""
    pending = list(inputs)
    written: list[str] = []

    def read(prompt: str) -> str:
        assert prompt == "cas12a [번호 | 34bp 타깃 서열]> "
        if not pending:
            raise EOFError
        return pending.pop(0)

    code = ln.run("cas12a", (), data_dir=data_dir, read=read, write=written.append)
    return code, written


def _once(data_dir, *arguments):
    written: list[str] = []
    errors: list[str] = []
    code = ln.run(
        "cas12a",
        arguments,
        data_dir=data_dir,
        write=written.append,
        write_error=errors.append,
    )
    return code, written, errors


@pytest.mark.parametrize("typed", ["A1", "report"])
def test_a_one_shot_call_prints_what_the_repl_prints(data_dir, typed):
    code, written, errors = _once(data_dir, typed)
    repl_code, repl_written = _repl(data_dir, [typed, "quit"])

    assert code == 0 and repl_code == 0 and not errors
    # The REPL first greets with its help, then prints the one screen.
    assert repl_written[1:] == written


def test_the_one_shot_screen_is_the_screen_module_s(data_dir):
    _, written, _ = _once(data_dir, "A1")
    expected = cas12a_screen.case_screen(rd.Cas12aReader(data_dir), "A1")
    assert written == [expected.rstrip("\n")]


def test_the_repl_greets_with_the_valid_numbers(data_dir):
    _, written = _repl(data_dir, [])
    assert "A1 low_error · C1 out_of_range" in written[0]
    assert "report" in written[0]


def test_an_unknown_number_names_the_range_and_fails_the_one_shot(data_dir):
    code, written, errors = _once(data_dir, "B3")
    assert code == 1 and not written
    assert errors == [
        "없는 사례 번호다: 'B3'. 유효 번호: A1 low_error · C1 out_of_range"
    ]


def test_an_unknown_number_keeps_the_repl_running(data_dir):
    code, written = _repl(data_dir, ["B3", "A1", "q"])
    assert code == 0
    assert written[1].startswith("없는 사례 번호다: 'B3'")
    assert written[2].startswith("A1  low_error")


def test_more_than_one_input_at_a_time_is_a_usage_message(data_dir):
    code, _, errors = _once(data_dir, "A1", "C1")
    assert code == 1
    assert "하나씩" in errors[0]


def test_spcas9_opens_its_own_screen(data_dir):
    """#173: the nuclease no longer reports itself as not ready."""
    written: list[str] = []
    errors: list[str] = []
    code = ln.run(
        "spcas9",
        ("B1",),
        data_dir=data_dir,
        write=written.append,
        write_error=errors.append,
    )

    assert code == 0 and not errors
    expected = spcas9_screen.case_screen(rd.SpCas9Reader(data_dir), "B1")
    assert written == [expected.rstrip("\n")]
    assert not any("준비 안 됨" in line for line in written)


def test_spcas9_opens_a_repl_that_names_its_own_input(data_dir):
    pending = ["B1", "quit"]
    written: list[str] = []

    def read(prompt: str) -> str:
        assert prompt == "spcas9 [번호 | 30bp 서열 (protospacer는 그 안)]> "
        return pending.pop(0)

    assert ln.run("spcas9", (), data_dir=data_dir, read=read, write=written.append) == 0
    assert written[0].startswith("9-Hydra v3")
    assert written[1].startswith("B1  GEN-ERRHI-01")


def test_the_two_nucleases_never_share_a_session(data_dir):
    """ADR 0012 made structural: a session holds one nuclease's reader only."""
    session = ln.Session(nuclease=ln.NUCLEASES["spcas9"], data_dir=data_dir)
    assert isinstance(session.reader, rd.SpCas9Reader)
    assert session.nuclease.screen is spcas9_screen


def test_main_routes_the_nuclease_and_spcas9_succeeds(capsys):
    assert entry.main(["spcas9", "report"]) == 0
    assert "9-Hydra v3" in capsys.readouterr().out


def test_main_without_arguments_prints_usage_naming_both_nucleases(capsys):
    assert entry.main([]) == 2
    usage = capsys.readouterr().err
    assert "cas12a" in usage and "spcas9" in usage
    assert "prepare" in usage and "verify" in usage


def test_main_with_an_unknown_nuclease_prints_usage(capsys):
    assert entry.main(["cas9"]) == 2
    assert "모르는 뉴클레아제" in capsys.readouterr().err


def test_main_no_longer_enters_the_old_menu_shell():
    """`shell.py` stays until #175 deletes it, but nothing launches it."""
    import ast
    import inspect

    imported = {
        alias.name
        for node in ast.walk(ast.parse(inspect.getsource(entry)))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "shell" not in imported and "launcher" in imported


#: Case A1's 34bp window in the fixture (source_row 9).
A1_SEQUENCE = "TGCATGCATGCATGCATGCATGCATGCATGCATG"


def test_each_prompt_states_its_own_input_unit():
    """The same word "서열" means two lengths, so each prompt names its own."""
    assert "34bp 타깃 서열" in ln.NUCLEASES["cas12a"].prompt
    assert ln.NUCLEASES["spcas9"].prompt.startswith("spcas9 [")
    assert "34bp" not in ln.NUCLEASES["spcas9"].prompt


def test_a_sequence_one_shot_prints_the_sequence_screen(data_dir):
    code, written, errors = _once(data_dir, A1_SEQUENCE)
    expected = cas12a_screen.sequence_screen(rd.Cas12aReader(data_dir), A1_SEQUENCE)
    assert code == 0 and not errors
    assert written == [expected.rstrip("\n")]


def test_a_sequence_in_the_repl_prints_what_the_one_shot_prints(data_dir):
    _, written, _ = _once(data_dir, A1_SEQUENCE)
    code, repl_written = _repl(data_dir, [A1_SEQUENCE, "quit"])
    assert code == 0 and repl_written[1:] == written


def test_a_sequence_not_stored_fails_the_one_shot_with_examples(data_dir):
    code, written, errors = _once(data_dir, "A" * 34)
    assert code == 1 and not written
    assert errors[0].startswith(f"저장된 3행에 없음: {'A' * 34}. 예: ")


def test_a_wrong_unit_names_both_inputs_and_keeps_the_repl_running(data_dir):
    code, written = _repl(data_dir, ["ACGT", "A1", "q"])
    assert code == 0
    assert written[1] == (
        "34bp 타깃 서열이 아니다: 길이 4 (34이어야 한다). "
        "사례 번호라면: A1 low_error · C1 out_of_range"
    )
    assert written[2].startswith("A1  low_error")


def _spcas9_once(data_dir, typed: str) -> tuple[int, list[str], list[str]]:
    written: list[str] = []
    errors: list[str] = []
    code = ln.run(
        "spcas9",
        (typed,),
        data_dir=data_dir,
        write=written.append,
        write_error=errors.append,
    )
    return code, written, errors


def test_a_30bp_sequence_one_shot_prints_the_spcas9_sequence_screen(data_dir):
    sequence = fx.spcas9_sequence(5)
    code, written, errors = _spcas9_once(data_dir, sequence)
    expected = spcas9_screen.screen_for(rd.SpCas9Reader(data_dir), sequence)
    assert code == 0 and not errors
    assert written == [expected.rstrip("\n")]
    assert written[0].startswith("서열 입력  row_index 5")


def test_a_20bp_protospacer_at_spcas9_names_the_unit_and_the_numbers(data_dir):
    code, written, errors = _spcas9_once(data_dir, fx.spcas9_sequence(5)[4:24])
    assert code == 1 and not written
    assert errors[0].startswith("30bp 서열이 아니다: 길이 20 (30이어야 한다) — ")
    assert errors[0].endswith("사례 번호라면: A1 ERRLO · B1 ERRHI · C2 HEADGAP")


def test_the_spcas9_help_names_the_sequence_input_and_its_population(data_dir):
    session = ln.Session(nuclease=ln.NUCLEASES["spcas9"], data_dir=data_dir)
    help_text = session.help_text()
    assert "30bp 서열" in help_text
    assert "generalisation 3행" in help_text


def test_the_help_names_the_sequence_input_and_its_population(data_dir):
    _, written = _repl(data_dir, [])
    assert "34bp 타깃 서열" in written[0]
    assert "저장된 3행" in written[0]
