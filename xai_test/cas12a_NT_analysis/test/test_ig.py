from types import SimpleNamespace

import numpy as np
import pytest

from xai_test.cas12a_NT_analysis.nt_inputs import sequence_hash
from xai_test.cas12a_NT_analysis.nt_predict import NativePredictor

# Synthetic raw-unit tolerances: not an approved real-NT quality policy.
POLICY = {
    "completeness_atol": 1e-5,
    "completeness_rtol": 1e-4,
    "near_zero": 1e-6,
    "convergence_atol": 1e-5,
    "convergence_rtol": 1e-4,
    "max_steps": 800,
}


def scalar_predictor(inputs, *, mode="cubic"):
    import torch
    from transformers import AutoTokenizer

    class ScalarEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4105, 8)
            self.bias = torch.nn.Parameter(torch.tensor([0.25]))
            self.config = SimpleNamespace(_attn_implementation="eager")
            with torch.no_grad():
                self.embedding.weight.fill_(0.5)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, attention_mask):
            # A nonconstant encoder + scalar head with a known cubic integral.
            values = self.embedding(input_ids)[:, :, 0].pow(3)
            if mode == "disconnected":
                values = values.detach()
            if mode == "zero":
                values = values * 0
            if mode == "nonfinite_gradient":
                values = values + (values - values.detach()).sqrt()
            logits = (values * attention_mask).sum(1) * 0.1 + self.bias
            return SimpleNamespace(logits=logits[:, None])

    return NativePredictor(
        ScalarEncoder().eval(),
        AutoTokenizer.from_pretrained(inputs.export, local_files_only=True),
        inputs.export,
    )


def source_row(predictor):
    sequence = "ACGT" + "TTTA" + "A" * 23 + "CGA"
    return {
        "source_row": 7,
        "sequence": sequence,
        "sequence_sha256": sequence_hash(sequence),
        "raw_prediction": float(predictor.predict([sequence])[0]),
    }


def test_cubic_ig_reconstructs_raw_output_including_cls_and_converges(inputs):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig

    predictor = scalar_predictor(inputs)
    result = compute_ig(source_row(predictor), predictor, policy=POLICY)
    assert result["accepted"] is True
    assert result["outputs"] == {
        "native": 0.375,
        "endpoint": 0.375,
        "reference": 0.25,
    }
    assert [attempt["steps"] for attempt in result["attempts"]] == [200, 400]
    final = result["summary"]
    np.testing.assert_allclose(final["dna_signed_mass"], [0.0125] * 9, atol=1e-6)
    np.testing.assert_allclose(final["special_signed_mass"], [0.0125, 0, 0], atol=1e-6)
    assert final["reconstructed_output"] == pytest.approx(0.375, abs=1e-6)
    assert final["quality"]["completeness_pass"] is True
    assert final["quality"]["convergence_pass"] is True
    predictor.assert_unchanged()
    assert all(parameter.grad is None for parameter in predictor.model.parameters())
    assert float(predictor.predict([result["row"]["sequence"]])[0]) == 0.375


@pytest.mark.parametrize(
    "mode,stage", [("disconnected", "gradient"), ("nonfinite_gradient", "gradient")]
)
def test_nonconstant_encoder_detects_gradient_defects_and_restores_native_path(
    inputs, mode, stage
):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig

    predictor = scalar_predictor(inputs, mode=mode)
    row = source_row(predictor)
    result = compute_ig(row, predictor, policy=POLICY)
    assert result["accepted"] is False
    assert result["failure"]["stage"] == stage
    assert result["failure"]["steps"] == 200
    assert result["arrays"]["signed_mass"].shape == (0, 12)
    assert result["summary"] is None
    assert float(predictor.predict([row["sequence"]])[0]) == row["raw_prediction"]


def test_connected_zero_attribution_is_valid_with_undefined_normalization(inputs):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig

    predictor = scalar_predictor(inputs, mode="zero")
    result = compute_ig(source_row(predictor), predictor, policy=POLICY)
    assert result["accepted"] is True
    assert result["summary"]["zero_norm"] is True
    assert result["summary"]["normalized_mass"] is None
    assert (
        result["summary"]["normalized_undefined_reason"] == "zero attribution L1 norm"
    )
    assert result["summary"]["quality"]["relative_residual"] is None
    assert (
        result["summary"]["quality"]["relative_undefined_reason"]
        == "near-zero output difference"
    )


def test_retry_limit_preserves_each_resolution_and_blocks_acceptance(inputs):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig

    predictor = scalar_predictor(inputs)
    strict = {key: (400 if key == "max_steps" else 0.0) for key in POLICY}
    result = compute_ig(
        source_row(predictor), predictor, policy=strict, integration_batch_size=71
    )
    assert result["accepted"] is False
    assert [attempt["steps"] for attempt in result["attempts"]] == [200, 400]
    assert result["attempts"][0]["batch_sizes"] == [71, 71, 58]
    assert result["failure"]["stage"] == "quality"
    assert result["summary"]["quality"]["completeness_pass"] is False
    assert result["summary"]["quality"]["convergence_pass"] is False


def test_native_esm_endpoint_preserves_positions_masks_and_token_dropout(inputs):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig

    predictor = NativePredictor.load(inputs.export)
    row = source_row(predictor)
    result = compute_ig(row, predictor, policy=POLICY, integration_batch_size=200)
    assert result["outputs"]["endpoint"] == pytest.approx(
        row["raw_prediction"], abs=1e-5, rel=1e-4
    )
    assert result["reference"]["attention_mask"] == [1] * 10 + [0, 0]
    assert result["reference"]["shape"] == [12, 8]
    assert result["failure"] is None or result["failure"]["stage"] == "quality"
    predictor.assert_unchanged()


def test_execution_faults_propagate_without_becoming_numerical_failures(inputs):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig

    predictor = scalar_predictor(inputs)
    row = source_row(predictor)

    def expired():
        raise ValueError("reservation expired")

    predictor.set_budget_guard(expired)
    with pytest.raises(ValueError, match="reservation expired"):
        compute_ig(row, predictor, policy=POLICY)
