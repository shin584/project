import numpy as np
import pytest

from xai_test.cas12a_NT_analysis.analyze import run_validation
from xai_test.cas12a_NT_analysis.nt_inputs import verify_inputs
from xai_test.cas12a_NT_analysis.nt_predict import NativePredictor
from xai_test.cas12a_NT_analysis.nt_run import Study
from xai_test.cas12a_NT_analysis.test.test_rollout import POLICY


def test_rollout_resumes_checked_sources_after_settlement_with_same_budget(
    inputs, tmp_path
):
    from xai_test.cas12a_NT_analysis.analyze import run_rollout
    from xai_test.cas12a_NT_analysis.nt_rollout_artifacts import read_rollout

    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    study = Study.create(tmp_path / "drive", fixture=True)
    predictions = run_validation(verified, study, "a", reserved_seconds=300)

    def interrupted_loader(*args, **kwargs):
        assert study.artifacts("a", kind="rollout-plan"), (
            "plan must be committed before model load"
        )
        predictor = NativePredictor.load(*args, **kwargs)
        original_guard = predictor.budget_guard

        def guard():
            original_guard()
            if study.artifacts("a", kind="rollout"):
                raise RuntimeError("interrupted after first source")

        predictor.set_budget_guard(guard)
        return predictor

    options = {"reserved_seconds": 300, "policy": POLICY}
    with pytest.raises(RuntimeError, match="interrupted"):
        run_rollout(verified, study, "a", loader=interrupted_loader, **options)
    partial = read_rollout(study, "a")
    assert partial["coverage"] == predictions["coverage"][:1]
    assert partial["job_unresolved"] is True
    first = partial["results"][0]["arrays"]["full"].copy()
    first_path, first_hash = next(
        (name, record["sha256"])
        for name, record in study.status()["runs"]["a"]["files"].items()
        if record["kind"] == "rollout"
    )
    reopened = Study.reopen(study.store.root, receipt=study.receipt)
    with pytest.raises(ValueError, match="unresolved"):
        run_rollout(verified, reopened, "a", **options)
    evidence = tmp_path / "terminated.txt"
    evidence.write_text("Fixture operator confirmed prior process termination")
    terminated = reopened.clock.utc()
    reopened.settle(
        evidence,
        terminated_utc=terminated,
        upper_seconds=terminated - reopened.status()["open_job"]["started_utc"] + 1,
    )
    final = run_rollout(verified, reopened, "a", **options)
    assert final["complete"] is True
    assert final["coverage"] == predictions["coverage"]
    assert final["gpu_executed"] is False
    np.testing.assert_array_equal(final["results"][0]["arrays"]["full"], first)
    assert reopened.status()["runs"]["a"]["files"][first_path]["sha256"] == first_hash
    assert len(reopened.artifacts("a", kind="rollout")) == 4
    assert reopened.status()["charged_seconds"] == 900

    def forbidden(*args, **kwargs):
        pytest.fail("reuse or reject before model loading")

    assert run_rollout(verified, reopened, "a", loader=forbidden, **options)["complete"]
    assert reopened.status()["charged_seconds"] == 900
    with pytest.raises(ValueError, match="plan"):
        run_rollout(
            verified,
            reopened,
            "a",
            loader=forbidden,
            **{**options, "policy": {"query_l1_threshold": 0.5}},
        )
    shard = next(reopened.store.root.glob("runs/a/artifacts/rollout-*.npz"))
    shard.write_bytes(b"corrupt previously complete fixture")
    with pytest.raises(ValueError, match="hash"):
        run_rollout(verified, reopened, "a", loader=forbidden, **options)


def test_driver_persists_attention_parity_failure_and_blocks_automatic_retry(
    inputs, tmp_path
):
    from xai_test.cas12a_NT_analysis.analyze import run_rollout
    from xai_test.cas12a_NT_analysis.nt_rollout_artifacts import read_rollout

    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    study = Study.create(tmp_path / "study", fixture=True)
    predictions = run_validation(verified, study, "a", reserved_seconds=300)

    def broken_loader(*args, **kwargs):
        predictor = NativePredictor.load(*args, **kwargs)

        def break_parity(module, arguments, options, output):
            if options.get("output_attentions") is True:
                output.logits = output.logits + 0.25

        predictor.model.register_forward_hook(break_parity, with_kwargs=True)
        return predictor

    with pytest.raises(ValueError, match="quality failure"):
        run_rollout(
            verified,
            study,
            "a",
            reserved_seconds=300,
            policy=POLICY,
            loader=broken_loader,
        )
    result = read_rollout(study, "a")
    assert result["failed"] == predictions["coverage"][:1]
    assert result["results"][0]["failure"]["stage"] == "parity"
    assert study.status()["open_job"] is None
    assert study.status()["charged_seconds"] == 600

    def forbidden(*args, **kwargs):
        pytest.fail("failed sources cannot automatically retry")

    with pytest.raises(ValueError, match="quality failure"):
        run_rollout(
            verified, study, "a", reserved_seconds=300, policy=POLICY, loader=forbidden
        )
