"""Evidence-backed final pre-fit freeze; no fitting during preparation."""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from SpCas9.spcas9_budget import BudgetLedger, configuration, environment_record, read
from SpCas9.spcas9_data import BASE_DIR
from SpCas9.spcas9_execution import reserve_destination, source_record
from SpCas9.spcas9_finalist import _seal, _sealed
from SpCas9.spcas9_split import file_sha256
from SpCas9.spcas9_study_inputs import StudyInputs, checked_bytes

# These exercise only the agreed study seam. The opt-in #83 learning checks
# are deliberately absent: their original cumulative allowance is exhausted.
REQUIRED_TESTS = (
    "test_spcas9_metrics.py",
    "test_spcas9_study.py::test_inputs_are_rechecked_before_backend_use_and_test_prediction_remains_closed",
    "test_spcas9_study.py::test_ngg_run_persists_selected_epoch_and_both_histories_or_refuses_invalid_evidence",
    "test_spcas9_ranking_checks.py",
    "test_spcas9_selection.py",
    "test_spcas9_budget.py::test_fixed_study_records_protocol_identity_and_blocks_premature_full_training",
    "test_spcas9_budget.py::test_six_fits_follow_fixed_order_and_cannot_reset_after_reopening",
    "test_spcas9_budget.py::test_failed_fit_consumes_attempt_and_blocks_all_later_fits",
    "test_spcas9_budget.py::test_changed_execution_identity_or_missing_history_refuses_training",
    "test_spcas9_budget.py::test_external_deadline_terminates_hung_backend_and_blocks_later_training",
    "test_spcas9_budget.py::test_synthetic_handoff_refuses_reset_and_additional_checks",
    "test_spcas9_budget.py::test_evaluation_and_bootstrap_do_not_consume_training_budget",
    "test_spcas9_acceptance.py",
    "test_spcas9_finalist.py",
    "test_spcas9_research.py",
    "test_spcas9_retry.py",
)
SYNTHETIC_EVIDENCE = BASE_DIR / "docs/evidence/issue83-synthetic-checks.json"
SYNTHETIC_SHA256 = "38bd4aa82cd60643a1d5d54d40fd61636e15611789c1e3cb54f7ba1f3f7232cd"


def _check_tests(path: Path) -> int:
    cases = ET.parse(path).findall(".//testcase")
    if not cases or any(
        c.tag in ("failure", "error", "skipped") for case in cases for c in case
    ):
        raise ValueError("readiness requires successful unskipped essential fixtures")
    covered = {case.attrib["classname"].split(".")[-1] + ".py" for case in cases}
    if covered != {test.split("::")[0] for test in REQUIRED_TESTS}:
        raise ValueError(
            "readiness fixture coverage differs from the required study checks"
        )
    return len(cases)


def verify_implementation(destination: Path) -> dict:
    destination = reserve_destination(Path(destination), (BASE_DIR / "SpCas9",))
    before, environment = source_record(), environment_record()
    command = [
        sys.executable,
        "-m",
        "pytest",
        *REQUIRED_TESTS,
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(destination / "fixtures"),
        "--junitxml",
        str(destination / "tests.xml"),
    ]
    with (destination / "tests.log").open("x", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=BASE_DIR, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode != 0:
        raise ValueError(f"essential fixtures failed; see {destination / 'tests.log'}")
    count = _check_tests(destination / "tests.xml")
    if source_record() != before or environment_record() != environment:
        raise ValueError("source/environment changed during readiness verification")
    record = {
        "issue": 88,
        "source": before,
        "environment": environment,
        "command": command,
        "passed": count,
        "files": {
            name: file_sha256(destination / name) for name in ("tests.xml", "tests.log")
        },
        "synthetic_checks": {
            "path": str(SYNTHETIC_EVIDENCE.resolve()),
            "sha256": SYNTHETIC_SHA256,
        },
    }
    checked_bytes(SYNTHETIC_EVIDENCE, SYNTHETIC_SHA256)
    _seal(record, destination, "readiness.json")
    return record


def _snapshot(study, readiness: Path, stored_checkpoint: Path) -> dict:
    config = study.execution_configuration()
    if config["mode"] != "full":
        raise PermissionError("final pre-fit readiness is only for the full execution")
    record = _sealed(readiness, "readiness.json")
    if (
        record["source"] != source_record()
        or record["environment"] != environment_record()
    ):
        raise ValueError("stale readiness source/environment")
    if (
        record["source"] != config["source"]
        or record["environment"] != config["environment"]
    ):
        raise ValueError("readiness differs from frozen execution identity")
    for name in ("tests.xml", "tests.log"):
        checked_bytes(readiness / name, record["files"][name])
    if record["passed"] != _check_tests(readiness / "tests.xml"):
        raise ValueError("readiness test count differs")
    checked_bytes(SYNTHETIC_EVIDENCE, SYNTHETIC_SHA256)
    synthetic = read(SYNTHETIC_EVIDENCE)
    ledger = config["synthetic"]
    checked_bytes(Path(ledger["path"]), ledger["sha256"])
    for key in ("sessions", "optimizer_updates_charged", "training_seconds_charged"):
        if synthetic[key] != ledger["record"][key]:
            raise ValueError("synthetic validation/accounting evidence differs")
    if "retry" in config:
        from SpCas9.spcas9_retry import verify_retry

        verify_retry(config, study.path)
    elif read(Path(ledger["path"] + ".study-owner.json")) != {
        "study": str(study.path.resolve()),
        "config_sha256": file_sha256(study.path / "execution-config.json"),
    }:
        raise ValueError("synthetic budget ownership differs")
    # Rebuilding the expected contract detects a re-sealed, modified recipe,
    # threshold, seed, budget or bootstrap configuration before the first fit.
    expected_config = configuration(config["protocol"], "full", ledger)
    if "retry" in config:
        expected_config["retry"] = config["retry"]
    if config != expected_config:
        raise ValueError("execution configuration differs from the fixed protocol")
    if config["protocol"]["inputs"] != StudyInputs().record():
        raise ValueError("full execution requires the pinned published inputs")
    StudyInputs().validate()
    checkpoint_hash = config["acceptance"]["historical_checkpoint_sha256"]
    checked_bytes(stored_checkpoint, checkpoint_hash)
    return {
        "issue": 88,
        "final_pre_fit_frozen": True,
        "execution_config_sha256": file_sha256(study.path / "execution-config.json"),
        "protocol_sha256": file_sha256(study.path / "protocol.json"),
        "source": config["source"],
        "environment": config["environment"],
        "readiness": {
            "path": str(readiness.resolve()),
            "sha256": file_sha256(readiness / "readiness.json"),
        },
        "synthetic_checks": {
            "path": str(SYNTHETIC_EVIDENCE.resolve()),
            "sha256": SYNTHETIC_SHA256,
        },
        "stored_checkpoint": str(stored_checkpoint.resolve()),
        "stored_checkpoint_sha256": checkpoint_hash,
        "exception": "#83 approved zero-update recovery; cumulative 395.993219800191 seconds remains charged",
    }


def freeze_execution(
    study, readiness: Path | None, stored_checkpoint: Path | None
) -> dict:
    if readiness is None or stored_checkpoint is None:
        raise PermissionError(
            "final pre-fit readiness evidence and stored checkpoint are required"
        )
    ledger = BudgetLedger(study.path)
    with ledger.locked():
        if ledger.status()["attempts"]:
            raise PermissionError("pre-fit freeze must precede all attempts")
        snapshot = _snapshot(study, Path(readiness), Path(stored_checkpoint))
        destination = reserve_destination(study.path / "pre-fit")
        _seal(snapshot, destination, "freeze.json")
    return snapshot


def verify_execution(study) -> dict:
    frozen = _sealed(study.path / "pre-fit", "freeze.json")
    readiness = Path(frozen["readiness"]["path"])
    checked_bytes(readiness / "readiness.json", frozen["readiness"]["sha256"])
    if frozen != _snapshot(study, readiness, Path(frozen["stored_checkpoint"])):
        raise ValueError("changed final pre-fit freeze")
    return frozen
