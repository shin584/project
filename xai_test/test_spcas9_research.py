"""Final study/report boundary; no real fits or additional optimizer checks."""

import json

import pytest
from test_spcas9_budget import controlled_inputs as _controlled_inputs
from test_spcas9_budget import synthetic_ledger as _synthetic_ledger
from test_spcas9_finalist import FinalistBackend
from test_spcas9_finalist import completed_study as _completed_study
from test_spcas9_study import inputs as _inputs

from SpCas9.spcas9_study import StudyProtocol

inputs = _inputs
controlled_inputs = _controlled_inputs
synthetic_ledger = _synthetic_ledger
completed_study = _completed_study


@pytest.fixture(autouse=True)
def short_report_bootstrap(monkeypatch):
    # Report classification is the behavior under test. Full 2000-draw quota,
    # cluster sampling and reproducibility are checked by acceptance fixtures.
    # This changes only these disposable verification-mode studies.
    from SpCas9.spcas9_acceptance import BOOTSTRAP_SETTINGS

    monkeypatch.setitem(BOOTSTRAP_SETTINGS, "valid_draws", 2)
    monkeypatch.setitem(BOOTSTRAP_SETTINGS, "maximum_attempts", 20)


def test_partial_report_keeps_six_run_statuses_and_independent_scanner_failure(
    controlled_inputs, synthetic_ledger, tmp_path
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    report = study.research_report(
        tmp_path / "report", scanner=tmp_path / "missing.json"
    )
    assert report["v2_a"]["status"] == "incomplete"
    assert report["v2_a"]["unresolved"] is True
    assert report["v2_b"]["status"] == "incomplete"
    assert report["implementation"]["status"] == "verification_only"
    assert len(report["planned_runs"]) == 6
    assert all(r["status"] == "not_started" for r in report["planned_runs"])
    assert report["budget"]["training_seconds"] == 30
    assert report["selected_checkpoint"] == "stored v2"
    assert report["supplementary_ci"]["status"] == "unavailable"
    assert (
        json.loads((tmp_path / "report/results.json").read_text())["v2_a"]
        == report["v2_a"]
    )
    with pytest.raises(FileExistsError):
        study.research_report(tmp_path / "report")


def test_completed_negative_test_is_a_result_and_damaged_evidence_is_incomplete(
    completed_study, tmp_path, monkeypatch
):
    study, validation, checkpoint = completed_study
    study.freeze_selection(validation, stored_checkpoint=checkpoint)
    test_dir = tmp_path / "test"
    study.evaluate_test(test_dir, backend=FinalistBackend(fail_test=True))
    report = study.research_report(
        tmp_path / "negative", validation=validation, test=test_dir
    )
    assert report["v2_a"] == {"status": "negative_test", "unresolved": True}
    assert report["selected_checkpoint"] == "stored v2"
    assert report["validation"]["decision"] == "pass"
    assert report["test"]["decision"] == "fail"
    assert report["v2_b"]["status"] == "incomplete"
    assert report["implementation"]["status"] == "verification_only"
    monkeypatch.setattr(
        "SpCas9.spcas9_finalist.source_record",
        lambda: {"revision": "later-report-recovery"},
    )
    recovered = study.research_report(
        tmp_path / "later", validation=validation, test=test_dir
    )
    assert recovered["v2_a"]["status"] == "negative_test"
    (study.path / "runs/candidate-42/predictions.csv").write_text("damaged")
    partial = study.research_report(
        tmp_path / "damaged", validation=validation, test=test_dir
    )
    assert partial["v2_a"]["status"] == "incomplete"
    assert partial["v2_a"]["unresolved"] is True


def test_prefit_freeze_requires_current_successful_implementation_evidence(
    inputs, synthetic_ledger, tmp_path
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        inputs,
        checkpoint_selection="ngg",
        execution_mode="full",
        synthetic_ledger=synthetic_ledger,
    )
    with pytest.raises(PermissionError, match="readiness"):
        study.freeze_execution()
    with pytest.raises((ValueError, FileNotFoundError, PermissionError)):
        study.freeze_execution(
            readiness=tmp_path / "absent", stored_checkpoint=tmp_path / "absent.keras"
        )
    assert study.budget_status()["attempts"] == []


def test_damaged_split_still_writes_incomplete_and_independent_scanner_report(
    controlled_inputs, synthetic_ledger, tmp_path
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    (controlled_inputs.split_dir / "val.csv").write_text("damaged required rows")
    report = study.research_report(tmp_path / "report")
    assert report["v2_a"]["status"] == "incomplete"
    assert "validation_reconstruction" in report["unavailable_evidence"]
    assert report["v2_b"]["status"] == "incomplete"
    assert (tmp_path / "report/results.json").is_file()


def test_corrupt_budget_remains_unknown_and_forbidden_while_scanner_is_reported(
    controlled_inputs, synthetic_ledger, tmp_path
):
    from SpCas9.spcas9_data import BASE_DIR

    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    journal = study.path / "budget.jsonl"
    journal.write_bytes(journal.read_bytes()[:-15])
    report = study.research_report(
        tmp_path / "report",
        scanner=BASE_DIR / "docs/evidence/issue81-scanner-v2-parity.json",
    )
    assert report["v2_a"]["status"] == "incomplete"
    assert report["budget"]["blocked"] is True
    assert report["budget"]["training_seconds"] is None
    assert all(run["status"] == "unknown" for run in report["planned_runs"])
    assert report["v2_b"]["status"] == "verified"
    assert (tmp_path / "report/results.json").is_file()
