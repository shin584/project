import subprocess
import sys

import pytest

from cas12a.nt_data import digest
from xai_test.cas12a_NT_analysis.nt_inputs import verify_inputs
from xai_test.cas12a_NT_analysis.nt_predict import NativePredictor
from xai_test.cas12a_NT_analysis.nt_validation import (
    ReceiptJournal,
    export_evidence,
    freeze_ig_coverage,
    freeze_protocol,
    open_study,
    run_phase,
)
from xai_test.cas12a_NT_analysis.nt_validation_protocol import prepare_protocol


def setup_validation(inputs, tmp_path):
    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    evidence = tmp_path / "evidence"
    study = open_study(tmp_path / "study", evidence, initialize=True, fixture=True)
    protocol = prepare_protocol(inputs.manifest, fixture=True)
    frozen = freeze_protocol(
        verified,
        study,
        inputs.manifest,
        protocol,
        evidence,
        approved_sha256=digest(protocol),
        single_executor=True,
        no_new_fp32_test=True,
    )
    return verified, study, evidence, frozen


def test_colab_public_workflow_replays_quality_freezes_and_reuses_without_loading(
    inputs, tmp_path
):
    verified, study, evidence, frozen = setup_validation(inputs, tmp_path)
    phases = list(frozen["protocol"]["groups"])
    for phase in phases:

        def loader(*args, phase=phase, **kwargs):
            # Every immutable acknowledgement must be independently usable.
            resumed = open_study(study.store.root, evidence, fixture=True)
            assert resumed.status()["open_job"] is not None
            predictor = NativePredictor.load(*args, **kwargs)
            original = predictor.parity

            def parity(sequences, **options):
                assert len(sequences) == len(frozen["protocol"]["groups"][phase])
                return original(sequences, **options)

            predictor.parity = parity
            return predictor

        result = run_phase(verified, study, frozen, phase, evidence, loader=loader)
        assert result["complete"] and result["accepted"]
        assert result["gpu_executed"] is False
        assert result["summary"]["peak_production_bytes"] is None
        assert result["summary"]["ig_seconds_by_source"]
    receipt = study.receipt
    resumed = open_study(study.store.root, evidence, fixture=True)

    def forbidden(*args, **kwargs):
        pytest.fail("completed phases must not load a model again")

    assert run_phase(verified, resumed, frozen, phases[-1], evidence, loader=forbidden)[
        "accepted"
    ]
    assert resumed.receipt == receipt
    decision = freeze_ig_coverage(resumed, frozen, inputs.manifest, evidence)
    assert decision["feasible"]
    assert decision["test_authorized"] is False
    assert decision["forecast"]["mode"] == "full"
    assert decision["selection_ids"] == sorted(inputs.partition["partitions"]["test"])
    # The existing #140/#142 interfaces must accept this intended Test identity
    # exactly; scientific protocol/code provenance lives in separate fields.
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import plan_ig
    from xai_test.cas12a_NT_analysis.nt_selection import (
        read_cases,
        read_coverage,
        save_cases,
    )

    coverage = read_coverage(
        evidence / "coverage", checksum=decision["coverage_sha256"]
    )
    source_by_id = {row["source_row"]: row for row in inputs.rows}
    test_rows = [
        {
            "source_row": i,
            "sequence": source_by_id[i]["sequence"],
            "label": source_by_id[i]["target"],
            "raw_prediction": 0.25,
        }
        for i in decision["storage_ids"]
    ]
    cases_root = tmp_path / "cases"
    case_sha = save_cases(
        evidence / "coverage",
        cases_root,
        coverage_checksum=decision["coverage_sha256"],
        predictions=test_rows,
        prediction_identity=decision["analysis_identity"],
    )
    cases = read_cases(
        evidence / "coverage",
        cases_root,
        coverage_checksum=decision["coverage_sha256"],
        checksum=case_sha,
    )
    prediction_identity = {
        **verified.identity,
        "partition": "test",
        "protocol": decision["test_prediction_protocol"],
    }
    plan = plan_ig(
        {
            "identity": prediction_identity,
            "rows": test_rows,
            "coverage": decision["storage_ids"],
            "coverage_complete": True,
        },
        run_id=decision["analysis_identity"]["run_id"],
        policy=frozen["protocol"]["config"]["ig_policy"],
        coverage=coverage,
        cases=cases,
    )
    assert plan["work"]["statistics_ids"] == decision["storage_ids"]
    assert freeze_ig_coverage(resumed, frozen, inputs.manifest, evidence) == decision
    archive = export_evidence(resumed, frozen, evidence, tmp_path / "returned.zip")
    assert archive["files"] > 20
    with pytest.raises(FileExistsError):
        export_evidence(resumed, frozen, evidence, tmp_path / "returned.zip")
    # The returned bundle must be independently usable with original assets
    # unavailable, and importing either model library forbidden.
    inputs.export.rename(tmp_path / "unavailable-export")
    code = """
import importlib.abc, json, sys
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise AssertionError('model import forbidden: ' + fullname)
sys.meta_path.insert(0, NoModel())
from xai_test.cas12a_NT_analysis.nt_validation import audit_evidence
result = audit_evidence(sys.argv[1], checksum=sys.argv[2], directory=sys.argv[3])
assert result['freeze']['feasible'] is True
assert result['complete'] is False and result['gpu_executed'] is False
assert len(result['phases']) == 4
print('model-free replay passed')
"""
    replay = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            archive["path"],
            archive["sha256"],
            str(tmp_path / "audit"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "model-free replay passed" in replay.stdout


def test_protocol_acceptance_and_phase_order_are_required_before_loading(
    inputs, tmp_path
):
    verified, study, evidence, frozen = setup_validation(inputs, tmp_path)
    with pytest.raises(ValueError, match="previous phase"):
        run_phase(verified, study, frozen, "cost-1", evidence)
    with pytest.raises(ValueError, match="exact displayed"):
        freeze_protocol(
            verified,
            study,
            inputs.manifest,
            frozen["protocol"],
            evidence,
            approved_sha256="0" * 64,
            single_executor=True,
            no_new_fp32_test=True,
        )
    with pytest.raises(ValueError, match="acknowledgements"):
        freeze_protocol(
            verified,
            study,
            inputs.manifest,
            frozen["protocol"],
            evidence,
            approved_sha256=frozen["protocol_sha256"],
        )


def test_receipts_are_external_and_never_rebuilt_from_ledger(inputs, tmp_path):
    _verified, study, evidence, frozen = setup_validation(inputs, tmp_path)
    latest = ReceiptJournal(evidence / "receipts", study.store.root).latest()
    assert latest == study.receipt
    receipt_path = (
        evidence / "receipts" / f"{latest['index']:08d}-{latest['head']}.json"
    )
    receipt_path.unlink()
    with pytest.raises(ValueError, match="missing or stale"):
        open_study(study.store.root, evidence, fixture=True)
    diagnostic = export_evidence(
        study, frozen, evidence, tmp_path / "failed-storage.zip"
    )
    assert diagnostic["history_verified"] is False
    with pytest.raises(ValueError, match="new and empty"):
        open_study(tmp_path / "another", evidence, initialize=True, fixture=True)
    with pytest.raises(ValueError, match="non-nested"):
        ReceiptJournal(study.store.root / "receipts", study.store.root)


def test_failed_receipt_write_blocks_model_load_and_poisons(inputs, tmp_path):
    verified, study, evidence, frozen = setup_validation(inputs, tmp_path)
    previous_sink = study.receipt_sink

    def failed_sink(receipt):
        if study.status()["open_job"]:
            raise OSError("external receipt storage failed")
        previous_sink(receipt)

    study.receipt_sink = failed_sink

    def forbidden(*args, **kwargs):
        pytest.fail("no GPU load before acknowledged start")

    with pytest.raises(OSError, match="receipt storage failed"):
        run_phase(verified, study, frozen, "correctness", evidence, loader=forbidden)
    assert study.poisoned
    with pytest.raises(ValueError, match="missing or stale"):
        open_study(study.store.root, evidence, fixture=True)


def test_partial_phase_requires_settlement_and_explicit_replacement(inputs, tmp_path):
    verified, study, evidence, frozen = setup_validation(inputs, tmp_path)

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated runtime termination")

    with pytest.raises(RuntimeError, match="runtime termination"):
        run_phase(verified, study, frozen, "correctness", evidence, loader=interrupted)
    resumed = open_study(study.store.root, evidence, fixture=True)
    with pytest.raises(ValueError, match="unresolved"):
        run_phase(verified, resumed, frozen, "correctness", evidence)
    note = tmp_path / "termination.txt"
    note.write_text("Operator independently confirmed previous runtime terminated.")
    terminated = resumed.clock.utc()
    resumed.settle(
        note,
        terminated_utc=terminated,
        upper_seconds=terminated - resumed.status()["open_job"]["started_utc"] + 1,
    )
    with pytest.raises(ValueError, match="replacement attempt"):
        run_phase(verified, resumed, frozen, "correctness", evidence)
    result = run_phase(verified, resumed, frozen, "correctness", evidence, attempt=1)
    assert result["accepted"]
    assert resumed.status()["charged_seconds"] == 600
    assert (
        open_study(study.store.root, evidence, fixture=True).receipt == resumed.receipt
    )


def test_saved_gradient_failure_blocks_following_and_replacement_phases(
    inputs, tmp_path
):
    from types import SimpleNamespace

    verified, study, evidence, frozen = setup_validation(inputs, tmp_path)

    def disconnected(*args, **kwargs):
        predictor = NativePredictor.load(*args, **kwargs)
        forward = predictor.model.forward

        def detached(**encoded):
            return SimpleNamespace(logits=forward(**encoded).logits.detach())

        predictor.model.forward = detached
        return predictor

    result = run_phase(
        verified, study, frozen, "correctness", evidence, loader=disconnected
    )
    assert result["complete"] is True and result["accepted"] is False
    resumed = open_study(study.store.root, evidence, fixture=True)
    with pytest.raises(ValueError, match="numerical failure"):
        run_phase(verified, resumed, frozen, "warmup", evidence)
    with pytest.raises(ValueError, match="numerical failure"):
        run_phase(verified, resumed, frozen, "correctness", evidence, attempt=1)
    with pytest.raises(ValueError, match="complete accepted"):
        freeze_ig_coverage(resumed, frozen, inputs.manifest, evidence)


def test_final_receipt_matches_closed_job_and_its_latency_is_budgeted(tmp_path):
    from xai_test.cas12a_NT_analysis.nt_run import Study
    from xai_test.cas12a_NT_analysis.test.test_run import Clock

    clock = Clock()
    saved = []

    def sink(receipt):
        if receipt["close_ack"]:
            saved.append(receipt)
            clock.seconds += 101

    study = Study.create(
        tmp_path / "study", clock=clock, fixture=True, receipt_sink=sink
    )
    study.register("a", {})
    study.start("a", "one", 100, {})
    clock.seconds += 1
    with pytest.raises(ValueError, match="overrun"):
        study.finish()
    assert saved[0]["close_ack"]["job_id"] == study.status()["last_closed"]["job_id"]
    assert study.poisoned
    assert study.status()["charged_seconds"] >= 102
    with pytest.raises(ValueError, match="overrun|missing or stale"):
        Study.reopen(study.store.root, receipt=saved[0], clock=clock)


@pytest.mark.parametrize("stage", ["parity", "ism"])
def test_native_numerical_failures_are_permanent_without_manual_settlement(
    inputs, tmp_path, stage
):
    import numpy as np

    verified, study, evidence, frozen = setup_validation(inputs, tmp_path)
    wild_type = next(
        r["sequence"]
        for r in verified.rows
        if r["source_row"] == frozen["protocol"]["groups"]["correctness"][0]
    )

    def numerical_loader(*args, **kwargs):
        predictor = NativePredictor.load(*args, **kwargs)
        if stage == "parity":

            def bad_parity(*args, **kwargs):
                raise ValueError("FP32 parity failure: singleton")

            predictor.parity = bad_parity
        else:
            forward = predictor.predict

            def invalid_mutant(sequences, **options):
                if any(sequence != wild_type for sequence in sequences):
                    return np.full(len(sequences), np.nan, dtype=np.float32)
                return forward(sequences, **options)

            predictor.predict = invalid_mutant
        return predictor

    result = run_phase(
        verified, study, frozen, "correctness", evidence, loader=numerical_loader
    )
    assert result["complete"] is True and result["accepted"] is False
    assert result["numerical_failure"] is True
    assert study.status()["open_job"] is None
    resumed = open_study(study.store.root, evidence, fixture=True)
    with pytest.raises(ValueError, match="numerical failure"):
        run_phase(verified, resumed, frozen, "correctness", evidence, attempt=1)
    with pytest.raises(ValueError, match="numerical failure"):
        run_phase(verified, resumed, frozen, "warmup", evidence)
