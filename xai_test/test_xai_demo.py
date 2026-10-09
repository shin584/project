"""Tests for the demo CLI's `demo` session and predictor loading (issue #27).

The default suite uses fake predictors and a stub/synthetic session only -
the NT regression checkpoint is never loaded. The single real-model smoke
test at the bottom is opt-in: set XAI_REAL_MODEL_TESTS=1 to run it.
"""

import functools
import json
import os
from pathlib import Path

import numpy as np
import pytest
import xai_demo
from model_b_testset_export import CASE_STUDY_MATCH_ATOL
from test_xai_session import (  # noqa: F401  (synthetic export fixture)
    NOVEL_SEQ,
    TESTSET_ONLY_SAMPLE,
    TESTSET_ONLY_SEQ,
    export_dir,
)
from xai_session import Predictors, XAISession, XAISessionError


def _scripted_input(monkeypatch, lines):
    """Feed `lines` to `input()`, then EOF; record the prompts shown."""
    remaining = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)


def _score_lines(text):
    """The `Model A` / `Model B` score lines of an `explain` header."""
    return [
        ln
        for ln in text.splitlines()
        if ln.startswith(("Model A ", "Model B ")) and " error " in ln
    ]


class StubSession:
    """Records which public operation each command reached."""

    def __init__(self):
        self.calls = []

    def cases(self):
        self.calls.append("cases")
        return "CASES-TEXT"

    def report(self):
        self.calls.append("report")
        return "REPORT-TEXT"

    def explain(self, query):
        self.calls.append(("explain", query))
        if query == "BOOM":
            raise RuntimeError("predictor crashed")
        if query == "SLOW":
            raise KeyboardInterrupt
        if query == "X99":
            raise XAISessionError("알 수 없는 Case Study ID: 'X99'")
        return f"EXPLAIN-{query}"


# ---------------------------------------------------------------------------
# Prompt loop
# ---------------------------------------------------------------------------


def test_session_dispatches_case_ids_cases_and_report(monkeypatch, capsys):
    session = StubSession()
    _scripted_input(monkeypatch, ["P01", "cases", "report", "quit"])
    xai_demo.run_session(session)
    out = capsys.readouterr().out
    assert session.calls == [("explain", "P01"), "cases", "report"]
    for text in ("EXPLAIN-P01", "CASES-TEXT", "REPORT-TEXT"):
        assert text in out


def test_an_exception_in_one_command_does_not_end_the_session(monkeypatch, capsys):
    session = StubSession()
    _scripted_input(monkeypatch, ["BOOM", "X99", "P01", "quit"])
    xai_demo.run_session(session)
    out = capsys.readouterr().out
    assert "error: RuntimeError: predictor crashed" in out
    assert "error: 알 수 없는 Case Study ID" in out
    assert "EXPLAIN-P01" in out  # still answering after both failures


def test_ctrl_c_during_a_command_returns_to_the_prompt(monkeypatch, capsys):
    session = StubSession()
    _scripted_input(monkeypatch, ["SLOW", "P01", "quit"])
    xai_demo.run_session(session)
    assert "EXPLAIN-P01" in capsys.readouterr().out


def test_help_lists_the_commands(monkeypatch, capsys):
    _scripted_input(monkeypatch, ["help", "quit"])
    xai_demo.run_session(StubSession())
    out = capsys.readouterr().out
    for command in ("cases", "report", "help", "quit", "P01"):
        assert command in out


def test_blank_lines_are_ignored_and_eof_ends_the_session(monkeypatch, capsys):
    session = StubSession()
    _scripted_input(monkeypatch, ["", "   ", "C01"])  # then EOF
    xai_demo.run_session(session)
    assert session.calls == [("explain", "C01")]


def test_a_typed_sequence_at_the_prompt_reaches_explain_unchanged(monkeypatch):
    session = StubSession()
    _scripted_input(monkeypatch, [NOVEL_SEQ.lower(), "quit"])
    xai_demo.run_session(session)
    assert session.calls == [("explain", NOVEL_SEQ.lower())]


def test_help_mentions_typed_sequences(monkeypatch, capsys):
    _scripted_input(monkeypatch, ["help", "quit"])
    xai_demo.run_session(StubSession())
    assert "36bp" in capsys.readouterr().out


def test_commands_are_case_insensitive(monkeypatch):
    session = StubSession()
    _scripted_input(monkeypatch, ["CASES", "Report", "QUIT", "P01"])
    xai_demo.run_session(session)
    assert session.calls == ["cases", "report"]  # nothing after QUIT


# ---------------------------------------------------------------------------
# Predictor loading
# ---------------------------------------------------------------------------


def test_predictors_load_one_at_a_time_with_progress_and_ready(capsys):
    events = []

    def loader(name):
        def load():
            events.append(f"start {name}")
            events.append(f"end {name}")
            return f"<{name}>"

        return load

    loaders = (
        ("Model A", "model_a", loader("a")),
        ("Model B", "model_b", loader("b")),
        ("Model B XAI", "model_b_xai", loader("xai")),
    )
    predictors = xai_demo.load_predictors(loaders)

    assert predictors == Predictors("<a>", "<b>", "<xai>")
    assert events == [
        "start a", "end a", "start b", "end b", "start xai", "end xai",
    ]  # fmt: skip
    out = capsys.readouterr().out
    assert "[1/3] Model A" in out and "[3/3] Model B XAI" in out
    assert "ready" in out.lower()


# ---------------------------------------------------------------------------
# CLI wiring: cached-only never constructs a predictor
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_cli(monkeypatch):
    """Route the CLI to a StubSession; record what predictors it was given."""
    state = {"loads": 0, "predictors": "unset", "session": StubSession()}

    def fake_load():
        state["loads"] += 1
        return Predictors("A", "B", "XAI")

    def fake_session(predictors=None, **kwargs):
        state["predictors"] = predictors
        state["session_kwargs"] = kwargs
        return state["session"]

    monkeypatch.setattr(xai_demo, "load_predictors", fake_load)
    monkeypatch.setattr(xai_demo, "XAISession", fake_session)
    return state


@pytest.mark.parametrize(
    "argv",
    [
        ["--cached", "demo"],
        ["demo", "--cached"],
        ["--cached", "explain", "P01"],
        ["cases"],
        ["report"],
    ],
)
def test_cached_and_model_free_commands_never_load_a_predictor(
    stub_cli, monkeypatch, argv
):
    _scripted_input(monkeypatch, ["quit"])
    assert xai_demo.main(argv) == 0
    assert stub_cli["loads"] == 0
    assert stub_cli["predictors"] is None


@pytest.mark.parametrize("argv", [["demo"], ["explain", "P01"]])
def test_live_commands_load_predictors_once_and_hand_them_to_the_session(
    stub_cli, monkeypatch, argv
):
    _scripted_input(monkeypatch, ["P01", "quit"])
    assert xai_demo.main(argv) == 0
    assert stub_cli["loads"] == 1
    assert stub_cli["predictors"] == Predictors("A", "B", "XAI")
    assert ("explain", "P01") in stub_cli["session"].calls


def test_a_model_load_failure_points_to_the_cached_fallback(
    stub_cli, monkeypatch, capsys
):
    def failing_load():
        raise MemoryError("out of memory")

    monkeypatch.setattr(xai_demo, "load_predictors", failing_load)
    assert xai_demo.main(["demo"]) == 1
    err = capsys.readouterr().err
    assert "MemoryError" in err and "--cached" in err


def test_cached_demo_never_calls_a_predictor_end_to_end(
    export_dir,  # noqa: F811
    monkeypatch,
    capsys,
):
    """Real session on the synthetic export, with predictor loading rigged to
    fail: `--cached demo` must still answer from the cache."""

    def no_load():
        raise AssertionError("--cached must never construct a predictor")

    monkeypatch.setattr(xai_demo, "load_predictors", no_load)
    monkeypatch.setattr(
        xai_demo,
        "XAISession",
        functools.partial(XAISession, export_dir, export_dir / "test_metadata.csv"),
    )
    _scripted_input(monkeypatch, ["P01", "cases", "quit"])
    assert xai_demo.main(["--cached", "demo"]) == 0
    out = capsys.readouterr().out
    assert "DISCORDANT_P01" in out
    model_lines = _score_lines(out)
    assert len(model_lines) == 2
    assert all("(cached)" in ln and "live" not in ln for ln in model_lines)


@pytest.mark.parametrize(
    "argv",
    [
        ["--ig-steps", "10", "explain", NOVEL_SEQ],
        ["explain", NOVEL_SEQ, "--ig-steps", "10"],
        ["--ig-steps", "10", "demo"],
    ],
)
def test_ig_steps_option_reaches_the_session(stub_cli, monkeypatch, argv):
    _scripted_input(monkeypatch, ["quit"])
    assert xai_demo.main(argv) == 0
    assert stub_cli["session_kwargs"] == {"ig_steps": 10}


def test_ig_steps_defaults_to_the_ig_module_default(stub_cli):
    assert xai_demo.main(["explain", NOVEL_SEQ]) == 0
    assert stub_cli["session_kwargs"] == {"ig_steps": 50}


def _cli_on_synthetic_export(monkeypatch, export_dir):  # noqa: F811
    def no_load():
        raise AssertionError("--cached must never construct a predictor")

    monkeypatch.setattr(xai_demo, "load_predictors", no_load)
    monkeypatch.setattr(
        xai_demo,
        "XAISession",
        functools.partial(XAISession, export_dir, export_dir / "test_metadata.csv"),
    )


def test_cached_explain_of_a_testset_sequence_answers_from_the_cache(
    export_dir,  # noqa: F811
    monkeypatch,
    capsys,
):
    _cli_on_synthetic_export(monkeypatch, export_dir)
    assert xai_demo.main(["--cached", "explain", TESTSET_ONLY_SEQ.lower()]) == 0
    out = capsys.readouterr().out
    assert f"sample_id {TESTSET_ONLY_SAMPLE}" in out
    assert "IG" in out and "Case Study" in out


def test_cached_explain_refuses_a_sequence_outside_the_testset(
    export_dir,  # noqa: F811
    monkeypatch,
    capsys,
):
    _cli_on_synthetic_export(monkeypatch, export_dir)
    assert xai_demo.main(["--cached", "explain", NOVEL_SEQ]) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "cached" in err


def test_an_invalid_sequence_prints_an_error_not_a_traceback(
    export_dir,  # noqa: F811
    monkeypatch,
    capsys,
):
    _cli_on_synthetic_export(monkeypatch, export_dir)
    assert xai_demo.main(["--cached", "explain", NOVEL_SEQ[:-1]]) == 1
    assert "36bp" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Opt-in real-model smoke test
# ---------------------------------------------------------------------------

REAL_MODELS = os.environ.get("XAI_REAL_MODEL_TESTS") == "1"


@pytest.mark.skipif(
    not REAL_MODELS,
    reason="loads Model A, Model B and the XAI predictor; set XAI_REAL_MODEL_TESTS=1",
)
def test_real_models_recompute_discordant_p01_within_tolerance():
    predictors = xai_demo.load_predictors()
    session = XAISession(predictors=predictors)

    summary_path = Path(xai_demo.__file__).parent / (
        "final_analysis_result/model_analysis_summary.json"
    )
    case = next(
        c
        for c in json.loads(summary_path.read_text(encoding="utf-8"))["case_studies"]
        if c["case_id"] == "DISCORDANT_P01"
    )
    live_a = predictors.model_a.predict([case["sequence"]])[0]
    live_b = predictors.model_b.predict([case["sequence"]])[0]
    # Same tolerance the session's match indicator uses.
    for live, cached in ((live_a, case["model_a"]), (live_b, case["model_b"])):
        np.testing.assert_allclose(live, cached["pred_raw"], atol=CASE_STUDY_MATCH_ATOL)

    text = session.explain("DISCORDANT_P01")
    model_lines = _score_lines(text)
    assert len(model_lines) == 2
    assert all("[일치]" in ln for ln in model_lines)
