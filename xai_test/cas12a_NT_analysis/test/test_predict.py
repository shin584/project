import numpy as np
import pytest

from xai_test.cas12a_NT_analysis.nt_inputs import verify_inputs
from xai_test.cas12a_NT_analysis.nt_predict import NativePredictor


def test_saved_scalar_fp32_native_batch_and_fresh_reload_parity(inputs):
    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    predictor = NativePredictor.load(verified.export_dir, device="cpu")
    sequences = [row["sequence"] for row in verified.rows]
    report = predictor.parity(sequences[:3], batch_size=2)
    assert report["validated"] is True
    assert report["actual_execution"]["output_dtype"] == "torch.float32"
    assert report["actual_execution"]["autocast"] is False
    assert report["actual_execution"]["tf32_matmul"] is False
    predictions = predictor.predict(sequences, batch_size=2)
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    predictor.model.train()
    with pytest.raises(ValueError, match="eval"):
        predictor.predict(sequences)


def test_inference_weight_mutation_is_rejected(inputs):
    import torch

    predictor = NativePredictor.load(inputs.export, device="cpu")
    with torch.no_grad():
        next(predictor.model.parameters()).add_(1)
    with pytest.raises(ValueError, match="weights/buffers changed"):
        predictor.assert_unchanged()


def test_parity_checks_budget_before_launching_another_forward(inputs):
    predictor = NativePredictor.load(inputs.export, device="cpu")
    observations = []

    def external_model_completed(module, arguments, output):
        observations.append("forward completed after reservation expired")

    def budget_guard():
        if observations:
            raise ValueError("GPU budget expired")

    predictor.model.register_forward_hook(external_model_completed)
    predictor.set_budget_guard(budget_guard)
    with pytest.raises(ValueError, match="budget expired"):
        predictor.parity([row["sequence"] for row in inputs.rows[:3]], batch_size=2)
    assert len(observations) == 1
