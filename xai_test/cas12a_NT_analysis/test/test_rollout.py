from types import SimpleNamespace

import numpy as np
import pytest

from xai_test.cas12a_NT_analysis.nt_inputs import sequence_hash
from xai_test.cas12a_NT_analysis.nt_predict import NativePredictor

# Synthetic diagnostic threshold, not scientific approval for real NT.
POLICY = {"query_l1_threshold": 1e-3}


def attention_predictor(inputs, *, mode="ordered"):
    import torch
    from transformers import AutoTokenizer

    class AttentionEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4105, 8)
            self.config = SimpleNamespace(
                _attn_implementation="eager", num_hidden_layers=2, num_attention_heads=2
            )
            self.esm = torch.nn.Module()
            self.esm.encoder = torch.nn.Module()
            self.esm.encoder.layer = torch.nn.ModuleList()
            for _ in range(2):
                layer = torch.nn.Module()
                layer.attention = torch.nn.Module()
                layer.attention.self = torch.nn.Identity()
                self.esm.encoder.layer.append(layer)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, attention_mask, output_attentions=False):
            logits = torch.full((len(input_ids), 1), 0.375)
            if output_attentions and mode == "parity":
                logits += 0.125
            matrices = []
            for index in range(2):
                matrix = torch.zeros(1, 2, 12, 12)
                matrix[:, :, :10, :10] = torch.eye(10)
                matrix[:, :, 10:, 0] = 1
                if mode == "uniform":
                    matrix[:, :, :, :10] = 0.1
                elif index == 0:
                    matrix[:, :, 1, :] = 0
                    matrix[:, :, 1, 0] = 1
                    matrix[:, :, 3, :] = 0
                    matrix[:, 0, 3, 0] = 1
                    matrix[:, 1, 3, 1] = 1
                else:
                    matrix[:, :, 2, :] = 0
                    matrix[:, :, 2, 1] = 1
                matrices.append(matrix)
            if mode == "nonfinite":
                matrices[0][0, 0, 1, 1] = float("nan")
            if mode == "shape":
                matrices[0] = matrices[0][:, :1]
            if mode == "layers":
                matrices.pop()
            if mode == "negative":
                matrices[0][0, 0, 1, 1] = -0.1
            if mode == "padding":
                matrices[0][0, 0, 1, 0] = 0
                matrices[0][0, 0, 1, 11] = 1
            if mode == "nonfinite_output" and output_attentions:
                logits.fill_(float("inf"))
            if mode == "missing":
                matrices = None
            return SimpleNamespace(logits=logits, attentions=matrices)

    return NativePredictor(
        AttentionEncoder().eval(),
        AutoTokenizer.from_pretrained(inputs.export, local_files_only=True),
        inputs.export,
    )


def rollout_row(predictor):
    sequence = "ACGT" + "TTTA" + "A" * 23 + "CGA"
    return {
        "source_row": 7,
        "sequence": sequence,
        "sequence_sha256": sequence_hash(sequence),
        "raw_prediction": float(predictor.predict([sequence])[0]),
    }


def test_rollout_preserves_cls_mass_and_noncommuting_layer_order(inputs):
    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout

    predictor = attention_predictor(inputs)
    result = compute_rollout(rollout_row(predictor), predictor, policy=POLICY)
    assert result["accepted"] is True
    arrays = result["arrays"]
    assert arrays["attention_mean"].shape == (2, 10, 10)
    assert arrays["full"].shape == (10, 10)
    np.testing.assert_array_equal(arrays["full"][2, :4], [0.25, 0.25, 0.5, 0])
    np.testing.assert_array_equal(arrays["full"][3, :4], [0.25, 0.25, 0, 0.5])
    np.testing.assert_array_equal(arrays["sequence"], arrays["full"][1:, 1:])
    assert result["summary"]["removed_cls_mass"][:2] == [0.5, 0.25]
    assert result["summary"]["sequence_row_mass"][:2] == [0.5, 0.75]
    assert result["summary"]["sequence_renormalized"] is False
    assert result["summary"]["full_query_diagnostic"]["status"] == "distinct_queries"
    assert result["tokens"]["valid_positions"] == list(range(10))
    assert result["tokens"]["dna_positions"] == list(range(1, 10))
    assert result["tokens"]["record"]["dna_spans"][-1] == [33, 34]
    predictor.assert_unchanged()


@pytest.mark.parametrize(
    "mode,stage",
    [
        ("parity", "parity"),
        ("missing", "attention"),
        ("nonfinite", "attention"),
        ("shape", "attention"),
        ("layers", "attention"),
        ("negative", "attention"),
        ("padding", "attention"),
        ("nonfinite_output", "attention_output"),
    ],
)
def test_attention_or_parity_defects_are_explicit_failed_diagnostics(
    inputs, mode, stage
):
    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout

    predictor = attention_predictor(inputs, mode=mode)
    result = compute_rollout(rollout_row(predictor), predictor, policy=POLICY)
    assert result["accepted"] is False
    assert result["failure"]["stage"] == stage
    assert result["failure"]["reason"]
    assert result["arrays"] is None
    assert result["summary"] is None
    assert result["outputs"]["native"] == 0.375


def test_degeneracy_uses_full_query_support_and_is_a_valid_limitation(inputs):
    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout

    predictor = attention_predictor(inputs, mode="uniform")
    result = compute_rollout(
        rollout_row(predictor), predictor, policy={"query_l1_threshold": 0.51}
    )
    diagnostic = result["summary"]["full_query_diagnostic"]
    assert result["accepted"] is True
    assert diagnostic["status"] == "near_identical_queries"
    assert diagnostic["l1_distances"] == pytest.approx([0.5] * 36)
    assert diagnostic["near_identical_fraction"] == 1
    assert diagnostic["query_pairs"][0] == [1, 2]


def test_saved_tiny_esm_returns_real_attention_probabilities_with_native_parity(inputs):
    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout

    predictor = NativePredictor.load(inputs.export)
    result = compute_rollout(rollout_row(predictor), predictor, policy=POLICY)
    assert result["accepted"] is True
    assert result["arrays"]["attention_mean"].shape == (1, 10, 10)
    assert result["execution"]["backend"] == "eager"
    assert result["execution"]["attention_classes"][0].endswith("EsmSelfAttention")
    assert result["quality"]["attention_parity"] is True
    assert result["quality"]["saved_prediction_parity"] is True
    predictor.assert_unchanged()


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"query_l1_threshold": -1},
        {"query_l1_threshold": float("nan")},
        {"query_l1_threshold": True},
        {"query_l1_threshold": 3},
    ],
)
def test_rollout_requires_explicit_valid_diagnostic_policy(inputs, policy):
    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout

    predictor = attention_predictor(inputs)
    with pytest.raises(ValueError, match="threshold"):
        compute_rollout(rollout_row(predictor), predictor, policy=policy)


def test_saved_prediction_parity_failure_and_budget_fault_are_distinct(inputs):
    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout

    predictor = attention_predictor(inputs)
    row = rollout_row(predictor)
    row["raw_prediction"] = 0.5
    result = compute_rollout(row, predictor, policy=POLICY)
    assert result["failure"]["stage"] == "parity"
    assert result["quality"]["saved_prediction_parity"] is False
    assert result["quality"]["attention_parity"] is True

    def interrupted():
        raise RuntimeError("budget interrupted")

    predictor.set_budget_guard(interrupted)
    with pytest.raises(RuntimeError, match="budget interrupted"):
        compute_rollout(row, predictor, policy=POLICY)
