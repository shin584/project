"""Tests for the XAI demo session (issue #23).

Drives only the session's public operations against a small synthetic
export directory built in the test - no real model or real export is read.
"""

import json

import numpy as np
import pandas as pd
import pytest
from xai_session import XAISession, XAISessionError

SEQ = "ACGTACGTACGTACGTACGTACGTACGGAGTACGTA"

CASES = [
    ("DISCORDANT_P01", "Primary Discordant", 3, 0.90, 0.55, 0.88),
    ("DISCORDANT_S01", "Secondary Discordant", 4, 0.70, 0.50, 0.62),
    ("DISCORDANT_R01", "Reverse Primary Discordant", 1, 0.40, 0.39, 0.61),
    ("DISCORDANT_RS01", "Reverse Secondary Discordant", 5, 0.30, 0.33, 0.45),
    ("CONCORDANT_C01", "Concordant", 0, 0.37, 0.371, 0.372),
]


def _case_entry(case_id, case_type, sample_id, true, pred_a, pred_b):
    return {
        "case_id": case_id,
        "sample_id": sample_id,
        "original_id": 100 + sample_id,
        "sequence": SEQ,
        "true_score": true,
        "model_a": {"pred_raw": pred_a, "error": abs(true - pred_a)},
        "model_b": {"pred_raw": pred_b, "error": abs(true - pred_b)},
        "case_type": case_type,
    }


@pytest.fixture
def export_dir(tmp_path):
    n = 6
    summary = {
        "metadata": {"total_samples": n},
        "global_evaluation": {},
        "complex_mismatch_metadata": [],
        "case_studies": [_case_entry(*c) for c in CASES],
    }
    (tmp_path / "model_analysis_summary.json").write_text(json.dumps(summary))
    np.savez(tmp_path / "model_analysis_arrays.npz", shap_values=np.zeros((n, 8964)))
    pd.DataFrame({"sample_id": range(n), "pred_raw": np.linspace(0.1, 0.9, n)}).to_csv(
        tmp_path / "model_b_testset_predictions.csv", index=False
    )
    pd.DataFrame(
        {
            "sample_id": range(n),
            "original_id": range(100, 100 + n),
            "sequence": [SEQ] * n,
            "true_score": np.linspace(0.2, 0.8, n),
            "sequence_hash": ["0" * 64] * n,
        }
    ).to_csv(tmp_path / "test_metadata.csv", index=False)
    return tmp_path


@pytest.fixture
def session(export_dir):
    return XAISession(export_dir, export_dir / "test_metadata.csv")


def test_session_without_predictors_is_cached_only(session):
    assert session.cached_only


def test_cases_lists_every_case_study_with_both_models(session):
    text = session.cases()
    for case_id, case_type, sample_id, true, pred_a, pred_b in CASES:
        line = next(ln for ln in text.splitlines() if case_id in ln)
        assert case_type in line
        assert str(sample_id) in line
        for value in (true, pred_a, abs(true - pred_a), pred_b, abs(true - pred_b)):
            assert f"{value:.3f}" in line
    assert "cached" in text


@pytest.mark.parametrize("query", ["P01", "p01", "DISCORDANT_P01", "discordant_p01"])
def test_explain_resolves_short_and_full_case_ids(session, query):
    text = session.explain(query)
    assert "DISCORDANT_P01" in text
    assert "0.900" in text  # true score
    assert "0.550" in text and "0.350" in text  # Model A pred / error
    assert "0.880" in text and "0.020" in text  # Model B pred / error


def test_explain_labels_scores_as_cached(session):
    text = session.explain("P01")
    model_lines = [
        ln for ln in text.splitlines() if ln.startswith(("Model A", "Model B"))
    ]
    assert model_lines
    assert all("cached" in ln for ln in model_lines)


@pytest.mark.parametrize(
    "query, case_type, marker",
    [
        ("P01", "Primary Discordant", "Model A"),
        ("S01", "Secondary Discordant", "Model A"),
        ("R01", "Reverse Primary Discordant", "Model B"),
        ("RS01", "Reverse Secondary Discordant", "Model B"),
        ("C01", "Concordant", "두 모델"),
    ],
)
def test_explain_prints_a_case_type_line(session, query, case_type, marker):
    text = session.explain(query)
    line = next(ln for ln in text.splitlines() if ln.startswith(f"{case_type}:"))
    assert marker in line


@pytest.mark.parametrize("query", ["P09", "X01", "DISCORDANT_P09", ""])
def test_explain_unknown_case_id_raises_clear_error(session, query):
    with pytest.raises(XAISessionError, match="Case Study"):
        session.explain(query)
