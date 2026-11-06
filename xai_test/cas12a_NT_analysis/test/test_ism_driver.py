import time

import numpy as np
import pytest

from xai_test.cas12a_NT_analysis.analyze import run_validation
from xai_test.cas12a_NT_analysis.nt_inputs import verify_inputs
from xai_test.cas12a_NT_analysis.nt_predict import NativePredictor
from xai_test.cas12a_NT_analysis.nt_run import Study


def test_driver_resumes_in_same_run_after_settlement_without_recomputing_shards(
    inputs, tmp_path
):
    from xai_test.cas12a_NT_analysis.analyze import run_ism
    from xai_test.cas12a_NT_analysis.nt_ism import read_ism

    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    study = Study.create(tmp_path / "drive", fixture=True)
    predictions = run_validation(
        verified, study, "a", reserved_seconds=300, batch_size=16
    )

    class Interrupted:
        def __init__(self, predictor):
            self.predictor = predictor
            self.calls = 0

        def __getattr__(self, name):
            return getattr(self.predictor, name)

        def predict(self, sequences, **options):
            self.calls += 1
            if self.calls == 8:
                raise RuntimeError("session interrupted during second source")
            return self.predictor.predict(sequences, **options)

    def interrupted_loader(*args, **options):
        assert study.status()["open_job"]["run_id"] == "a"
        return Interrupted(NativePredictor.load(*args, **options))

    with pytest.raises(RuntimeError, match="session interrupted"):
        run_ism(
            verified,
            study,
            "a",
            reserved_seconds=300,
            batch_size=16,
            loader=interrupted_loader,
        )
    partial = read_ism(study, "a")
    assert partial["coverage"] == predictions["coverage"][:1]
    assert partial["job_unresolved"] is True
    first_arrays = partial["arrays"]["prediction"].copy()
    reopened = Study.reopen(study.store.root, receipt=study.receipt)
    with pytest.raises(ValueError, match="unresolved"):
        run_ism(verified, reopened, "a", reserved_seconds=300, batch_size=16)
    evidence = tmp_path / "terminated.txt"
    evidence.write_text("Fixture operator confirmed prior session termination.")
    terminated = time.time()
    reopened.settle(
        evidence,
        terminated_utc=terminated,
        upper_seconds=terminated - reopened.status()["open_job"]["started_utc"] + 1,
    )
    result = run_ism(verified, reopened, "a", reserved_seconds=300, batch_size=16)
    assert result["complete"] is True
    assert result["mutant_count"] == 408
    assert result["coverage"] == predictions["coverage"]
    assert result["gpu_executed"] is False
    np.testing.assert_array_equal(result["arrays"]["prediction"][:1], first_arrays)
    assert len(reopened.artifacts("a", kind="ism")) == 4
    assert reopened.status()["charged_seconds"] == 900

    def forbidden(*args, **kwargs):
        pytest.fail("completed ISM coverage must be reused without loading")

    complete = Study.reopen(reopened.store.root, receipt=reopened.receipt)
    same = run_ism(
        verified, complete, "a", reserved_seconds=300, batch_size=16, loader=forbidden
    )
    np.testing.assert_array_equal(
        same["arrays"]["prediction"], result["arrays"]["prediction"]
    )
    assert complete.status()["charged_seconds"] == 900
    with pytest.raises(ValueError, match="identity mismatch"):
        run_ism(
            verified,
            complete,
            "a",
            reserved_seconds=300,
            batch_size=8,
            loader=forbidden,
        )
    (inputs.export / "config.json").write_bytes(b"changed synthetic export")
    with pytest.raises(ValueError, match="export hash"):
        run_ism(
            verified,
            complete,
            "a",
            reserved_seconds=300,
            batch_size=16,
            loader=forbidden,
        )
