import numpy as np
import pytest

from xai_test.cas12a_NT_analysis.analyze import run_validation
from xai_test.cas12a_NT_analysis.nt_inputs import verify_inputs
from xai_test.cas12a_NT_analysis.nt_predict import NativePredictor
from xai_test.cas12a_NT_analysis.nt_run import Study

# Deliberately permissive CPU fixture policy, not a real NT quality approval.
DRIVER_POLICY = {
    "completeness_atol": 0.1,
    "completeness_rtol": 0.1,
    "near_zero": 1e-6,
    "convergence_atol": 0.1,
    "convergence_rtol": 0.1,
    "max_steps": 400,
}


def test_driver_reuses_ig_sources_after_interruption_and_retains_budget(
    inputs, tmp_path
):
    from xai_test.cas12a_NT_analysis.analyze import run_ig
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import read_ig

    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    study = Study.create(tmp_path / "drive", fixture=True)
    predictions = run_validation(
        verified, study, "a", reserved_seconds=300, batch_size=16
    )
    ids = predictions["coverage"][:2]

    def interrupted_loader(*args, **kwargs):
        predictor = NativePredictor.load(*args, **kwargs)
        original_guard = predictor.budget_guard

        def guard():
            original_guard()
            if len(study.status()["runs"]["a"]["files"]) and study.artifacts(
                "a", kind="ig"
            ):
                raise RuntimeError("session interrupted after first source")

        predictor.set_budget_guard(guard)
        return predictor

    options = {
        "reserved_seconds": 300,
        "batch_size": 16,
        "policy": DRIVER_POLICY,
        "integration_batch_size": 200,
        "probe_ids": ids,
    }
    with pytest.raises(RuntimeError, match="session interrupted"):
        run_ig(verified, study, "a", loader=interrupted_loader, **options)
    partial = read_ig(study, "a")
    assert partial["coverage"] == ids[:1]
    assert partial["job_unresolved"] is True
    first = partial["results"][0]["arrays"]["signed_mass"].copy()
    reopened = Study.reopen(study.store.root, receipt=study.receipt)
    with pytest.raises(ValueError, match="unresolved"):
        run_ig(verified, reopened, "a", **options)
    evidence = tmp_path / "terminated.txt"
    evidence.write_text("Fixture operator confirmed process termination")
    terminated = reopened.clock.utc()
    reopened.settle(
        evidence,
        terminated_utc=terminated,
        upper_seconds=terminated - reopened.status()["open_job"]["started_utc"] + 1,
    )
    final = run_ig(verified, reopened, "a", **options)
    assert final["complete"] is True
    assert final["coverage"] == ids
    assert final["gpu_executed"] is False
    np.testing.assert_array_equal(final["results"][0]["arrays"]["signed_mass"], first)
    assert reopened.status()["charged_seconds"] == 900

    def forbidden(*args, **kwargs):
        pytest.fail("must reject or reuse before loading")

    same = run_ig(verified, reopened, "a", loader=forbidden, **options)
    assert same["complete"] is True
    assert reopened.status()["charged_seconds"] == 900
    with pytest.raises(ValueError, match="plan"):
        run_ig(
            verified,
            reopened,
            "a",
            loader=forbidden,
            **{**options, "probe_ids": predictions["coverage"][2:]},
        )
    with pytest.raises(ValueError, match="plan"):
        run_ig(
            verified,
            reopened,
            "a",
            loader=forbidden,
            **{**options, "policy": {**DRIVER_POLICY, "near_zero": 0}},
        )


def test_driver_saves_gradient_failure_and_blocks_automatic_retry(inputs, tmp_path):
    from types import SimpleNamespace

    from xai_test.cas12a_NT_analysis.analyze import run_ig
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import read_ig

    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    study = Study.create(tmp_path / "drive", fixture=True)
    predictions = run_validation(
        verified, study, "a", reserved_seconds=300, batch_size=16
    )

    def disconnected_loader(*args, **options):
        predictor = NativePredictor.load(*args, **options)
        native_forward = predictor.model.forward

        def disconnected(**encoded):
            return SimpleNamespace(logits=native_forward(**encoded).logits.detach())

        predictor.model.forward = disconnected
        return predictor

    options = {
        "reserved_seconds": 300,
        "batch_size": 16,
        "policy": DRIVER_POLICY,
        "integration_batch_size": 200,
        "probe_ids": predictions["coverage"][:1],
    }
    with pytest.raises(ValueError, match="quality failure"):
        run_ig(verified, study, "a", loader=disconnected_loader, **options)
    report = read_ig(study, "a")
    assert report["failed"] == predictions["coverage"][:1]
    assert report["results"][0]["failure"]["stage"] == "gradient"
    assert report["complete"] is False
    assert report["job_unresolved"] is False
    assert study.status()["charged_seconds"] == 600

    def forbidden(*args, **options):
        pytest.fail("failed plan must be rejected before model loading")

    with pytest.raises(ValueError, match="quality failure"):
        run_ig(verified, study, "a", loader=forbidden, **options)
    assert study.status()["charged_seconds"] == 600
