"""The new launcher: the nuclease is the first argument and fixes the session.

`cas12a A1` and typing `A1` at the `cas12a>` prompt go through the same
`Session.execute`, so what the meeting sees cannot drift from a one-shot run.
"""

from __future__ import annotations

import pytest

from xai_test.nt_xai_demo import __main__ as entry
from xai_test.nt_xai_demo import cas12a_screen
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
        assert prompt == "cas12a> "
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


def test_spcas9_says_not_ready_and_exits(data_dir):
    written: list[str] = []
    code = ln.run("spcas9", ("B1",), data_dir=data_dir, write=written.append)
    assert code == 1
    assert len(written) == 1
    assert "준비 안 됨" in written[0] and "#173" in written[0]


def test_spcas9_does_not_open_a_repl(data_dir):
    def read(prompt: str) -> str:
        raise AssertionError("spcas9 must not prompt")

    assert ln.run("spcas9", (), data_dir=data_dir, read=read, write=lambda _: None) == 1


def test_main_routes_the_nuclease_and_spcas9_exits_non_zero(capsys):
    assert entry.main(["spcas9"]) == 1
    assert "준비 안 됨" in capsys.readouterr().err


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
