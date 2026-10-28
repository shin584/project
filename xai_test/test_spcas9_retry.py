"""Explicit one-time retry through the study boundary; fake backends only."""

import json

import pytest
from test_spcas9_budget import FailingBackend, FixedBackend
from test_spcas9_budget import controlled_inputs as _controlled_inputs
from test_spcas9_budget import synthetic_ledger as _synthetic_ledger
from test_spcas9_study import inputs as _inputs

from SpCas9.spcas9_study import StudyProtocol

inputs = _inputs
controlled_inputs = _controlled_inputs
synthetic_ledger = _synthetic_ledger


@pytest.fixture
def stopped_study(controlled_inputs, synthetic_ledger, tmp_path):
    study = StudyProtocol.plan(
        tmp_path / "original",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    run = study.plan_run(recipe="baseline", seed=42)
    with pytest.raises(RuntimeError):
        study.train_validation(run, backend=FailingBackend("failure"))
    return study


def test_authorized_retry_carries_history_once_and_preserves_parent(
    stopped_study, controlled_inputs, synthetic_ledger, tmp_path
):
    parent = stopped_study
    original = (parent.path / "budget.jsonl").read_bytes()
    previous = parent.budget_status()
    retry = StudyProtocol.plan(
        tmp_path / "retry",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
        retry_from=parent.path,
        retry_authorization="User explicitly requested retry of V2-A",
    )
    assert retry.budget_status()["training_seconds"] == previous["training_seconds"]
    config = retry.execution_configuration()
    assert config["retry"]["aggregate_attempt_cap"] == 7
    assert config["retry"]["parent_attempts"] == previous["attempts"]
    assert len(config["run_order"]) == 6
    run = retry.plan_run(recipe="baseline", seed=42)
    assert retry.train_validation(run, backend=FixedBackend())["eligible"]
    assert retry.budget_status()["training_seconds"] > previous["training_seconds"]
    assert (parent.path / "budget.jsonl").read_bytes() == original
    assert parent.budget_status()["blocked"] is True
    with pytest.raises((FileExistsError, ValueError), match="retry|reserved"):
        StudyProtocol.plan(
            tmp_path / "duplicate",
            controlled_inputs,
            checkpoint_selection="ngg",
            execution_mode="verification",
            synthetic_ledger=synthetic_ledger,
            retry_from=parent.path,
            retry_authorization="duplicate",
        )


def test_retry_requires_explicit_authorization_and_intact_prior_accounting(
    stopped_study, controlled_inputs, synthetic_ledger, tmp_path
):
    options = {
        "checkpoint_selection": "ngg",
        "execution_mode": "verification",
        "synthetic_ledger": synthetic_ledger,
        "retry_from": stopped_study.path,
    }
    with pytest.raises(ValueError, match="authorization"):
        StudyProtocol.plan(tmp_path / "unauthorized", controlled_inputs, **options)
    journal = stopped_study.path / "budget.jsonl"
    journal.write_bytes(journal.read_bytes()[:-10])
    with pytest.raises(ValueError):
        StudyProtocol.plan(
            tmp_path / "damaged",
            controlled_inputs,
            retry_authorization="explicit retry",
            **options,
        )
    assert not (stopped_study.path / "authorized-retry.json").exists()


def test_failure_in_authorized_retry_still_stops_all_later_fits(
    stopped_study, controlled_inputs, synthetic_ledger, tmp_path
):
    retry = StudyProtocol.plan(
        tmp_path / "retry",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
        retry_from=stopped_study.path,
        retry_authorization="User explicitly requested retry of V2-A",
    )
    run = retry.plan_run(recipe="baseline", seed=42)
    with pytest.raises(RuntimeError):
        retry.train_validation(run, backend=FailingBackend("failure"))
    next_run = retry.plan_run(recipe="baseline", seed=43)
    with pytest.raises(ValueError, match="blocks"):
        retry.train_validation(next_run, backend=FixedBackend())
    assert len(retry.budget_status()["attempts"]) == 1
    assert retry.budget_status()["blocked"] is True
    receipt = json.loads((stopped_study.path / "authorized-retry.json").read_text())
    assert receipt["study"] == str(retry.path.resolve())


def test_parent_damage_blocks_retry_identity_but_keeps_partial_scanner_report(
    stopped_study, controlled_inputs, synthetic_ledger, tmp_path
):
    from SpCas9.spcas9_data import BASE_DIR

    retry = StudyProtocol.plan(
        tmp_path / "retry",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
        retry_from=stopped_study.path,
        retry_authorization="User explicitly requested retry of V2-A",
    )
    journal = stopped_study.path / "budget.jsonl"
    journal.write_bytes(journal.read_bytes()[:-10])
    with pytest.raises(ValueError, match="history|journal|JSON"):
        retry.execution_configuration()
    report = retry.research_report(
        tmp_path / "report",
        scanner=BASE_DIR / "docs/evidence/issue88-scanner-v2-parity.json",
    )
    assert report["v2_a"]["status"] == "incomplete"
    assert report["budget"]["blocked"] is True
    assert report["v2_b"]["status"] == "verified"
