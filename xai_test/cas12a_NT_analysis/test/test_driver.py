import pytest

from xai_test.cas12a_NT_analysis.analyze import run_validation
from xai_test.cas12a_NT_analysis.nt_inputs import verify_inputs
from xai_test.cas12a_NT_analysis.nt_predict import NativePredictor
from xai_test.cas12a_NT_analysis.nt_run import Study


def test_driver_loads_only_after_durable_start_exports_and_reuses_complete_run(
    inputs, tmp_path
):
    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    root = tmp_path / "drive"
    study = Study.create(root, fixture=True)

    def load_after_start(export, *, device, **options):
        observed = Study.reopen(root, receipt=study.receipt)
        assert observed.status()["open_job"]["run_id"] == "validation"
        return NativePredictor.load(export, device=device, **options)

    report = run_validation(
        verified,
        study,
        "validation",
        reserved_seconds=300,
        batch_size=2,
        loader=load_after_start,
    )
    assert report["complete"] is True
    assert report["coverage"] == sorted(inputs.partition["partitions"]["validation"])
    assert report["gpu_executed"] is False
    assert study.status()["charged_seconds"] == 300
    reopened = Study.reopen(root, receipt=study.receipt)

    def should_not_load(*args, **kwargs):
        pytest.fail("complete validated coverage must be reused without model loading")

    same = run_validation(
        verified,
        reopened,
        "validation",
        reserved_seconds=300,
        batch_size=2,
        loader=should_not_load,
    )
    assert same == report


def test_partial_run_resumes_only_after_evidence_and_reuses_committed_rows(
    inputs, tmp_path
):
    import time

    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    root = tmp_path / "drive"
    study = Study.create(root, fixture=True)

    class InterruptedModel:
        def __init__(self, model):
            self.model = model
            self.calls = 0

        def __getattr__(self, name):
            return getattr(self.model, name)

        def predict(self, sequences, **options):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated session interruption")
            return self.model.predict(sequences, **options)

    def interrupted_loader(export, *, device, **options):
        return InterruptedModel(NativePredictor.load(export, device=device, **options))

    with pytest.raises(RuntimeError, match="session interruption"):
        run_validation(
            verified,
            study,
            "a",
            reserved_seconds=300,
            batch_size=2,
            loader=interrupted_loader,
        )
    from xai_test.cas12a_NT_analysis.nt_artifacts import read_results

    partial = read_results(study, "a")
    assert partial["coverage"] == verified.identity["validation_source_rows"][:2]
    assert partial["job_unresolved"] is True
    reopened = Study.reopen(root, receipt=study.receipt)
    with pytest.raises(ValueError, match="unresolved"):
        run_validation(verified, reopened, "a", reserved_seconds=300, batch_size=2)
    evidence = tmp_path / "termination.txt"
    evidence.write_text("External operator confirmed previous process termination.")
    terminated = time.time()
    duration = terminated - reopened.status()["open_job"]["started_utc"] + 1
    reopened.settle(evidence, terminated_utc=terminated, upper_seconds=duration)
    complete = run_validation(
        verified, reopened, "a", reserved_seconds=300, batch_size=2
    )
    assert complete["coverage"] == verified.identity["validation_source_rows"]
    assert complete["complete"] is True
    assert reopened.status()["charged_seconds"] == 600
    assert len(reopened.artifacts("a")) == 2


def test_start_storage_failure_prevents_model_loading(inputs, tmp_path):
    import json

    from xai_test.cas12a_NT_analysis.nt_run import DirectoryStore

    class StartFailureStore(DirectoryStore):
        def put(self, name, data):
            if (
                name.startswith("ledger/")
                and json.loads(data)["payload"]["event"] == "start-before-model-load"
            ):
                raise OSError("Drive start record failed")
            return super().put(name, data)

    store = StartFailureStore(tmp_path / "drive")
    study = Study.create(store.root, fixture=True, store=store)
    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )

    def forbidden_loader(*args, **kwargs):
        pytest.fail("model must not be loaded without verified durable start")

    with pytest.raises(OSError, match="start record"):
        run_validation(
            verified,
            study,
            "a",
            reserved_seconds=300,
            batch_size=2,
            loader=forbidden_loader,
        )
    assert study.poisoned is True


def test_checkpoint_changed_since_verification_is_rejected_before_loading(
    inputs, tmp_path
):
    from safetensors.torch import load_file, save_file

    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    weights = inputs.export / "model.safetensors"
    state = {name: tensor.clone() for name, tensor in load_file(weights).items()}
    state["classifier.out_proj.bias"] = state["classifier.out_proj.bias"] + 1
    save_file(state, weights)
    study = Study.create(tmp_path / "drive", fixture=True)

    def forbidden_loader(*args, **kwargs):
        pytest.fail("changed checkpoint must fail before model loading")

    with pytest.raises(ValueError, match="export hash"):
        run_validation(
            verified, study, "a", reserved_seconds=300, loader=forbidden_loader
        )
