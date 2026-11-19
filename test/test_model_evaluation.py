"""#202 metric and parity boundaries: hand-worked examples, no model mocks."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "docs/model-evaluation"
SPEC = importlib.util.spec_from_file_location("model_evaluation", DIRECTORY / "evaluate.py")
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
POLICY = json.loads((DIRECTORY / "contract.json").read_text())["parity"]


def test_missing_labels_are_excluded_per_head_but_all_predictions_must_be_finite():
    metrics = evaluation.performance([0, np.nan, 1], [0.25, 0.75, 0.75])
    assert (metrics["n"], metrics["missing"]) == (2, 1)
    assert metrics["mae"] == 0.25
    assert metrics["mse"] == 0.0625
    assert metrics["pearson"] == pytest.approx(1)
    assert metrics["spearman"] == pytest.approx(1)
    with pytest.raises(ValueError, match="nonfinite prediction"):
        evaluation.performance([0, np.nan, 1], [0.25, np.nan, 0.75])


def test_tied_ranks_and_error_metrics_have_independently_known_values():
    metrics = evaluation.performance([0, 0, 1, 1], [0, 1, 2, 3])
    assert metrics["mae"] == 1
    assert metrics["mse"] == 1.5
    assert metrics["pearson"] == pytest.approx(2 / np.sqrt(5))
    assert metrics["spearman"] == pytest.approx(2 / np.sqrt(5))


@pytest.mark.parametrize("target,prediction", [([], []), ([1], [2]), ([1, 1], [1, 2]), ([0, 1], [2, 2])])
def test_undefined_correlations_remain_null_with_a_reason(target, prediction):
    metrics = evaluation.performance(target, prediction)
    for name in ("pearson", "spearman"):
        assert metrics[name] is None
        assert metrics["unavailable_reasons"][name]


def test_clipped_agreement_cannot_hide_a_raw_output_parity_failure():
    result = evaluation.compare([0, 1], [-0.2, 1.2], [-0.3, 1.3], POLICY)
    assert result["passed"] is False
    assert result["output_passed"] is False
    assert result["metrics"]["raw"]["reference"]["mae"] == pytest.approx(0.2)
    assert result["metrics"]["clipped"]["reference"]["mae"] == 0
    assert result["metrics"]["clipped"]["observed"]["mae"] == 0


def test_per_output_failure_is_not_averaged_away():
    expected = np.zeros(1500)
    observed = expected.copy()
    observed[1499] = 0.00001
    assert evaluation.compare(expected, expected, observed, POLICY)["passed"] is False


def test_near_tie_metric_failure_is_kept_even_when_output_parity_passes():
    result = evaluation.compare([0, 1, 2], [0, 1e-8, 2e-8], [0, 2e-8, 1e-8], POLICY)
    assert result["output_passed"] is True
    assert result["passed"] is False
    assert result["metrics"]["raw"]["differences"]["spearman"]["passed"] is False


def test_registered_file_mutation_is_rejected(tmp_path):
    (tmp_path / "inputs").write_bytes(b"frozen inputs")
    digest = evaluation.sha(tmp_path / "inputs")
    evaluation.write(tmp_path / "registration.json", {"files": {"inputs": digest}})
    evaluation.verify_inventory(tmp_path)
    (tmp_path / "inputs").write_bytes(b"different inputs")
    with pytest.raises(ValueError, match="frozen input changed"):
        evaluation.verify_inventory(tmp_path)
