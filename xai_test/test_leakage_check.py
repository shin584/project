import json

import numpy as np
import pandas as pd
import pytest
from leakage_check import (
    MIN_N_FOR_SPEARMAN,
    build_report,
    case_study_membership,
    compare_abs_errors,
    group_metrics,
    label_membership,
)


def test_label_membership_flags_each_old_split_separately_and_normalizes_case():
    sequences = pd.Series(["AAAA", "cccc", "GGGG", "TTTT"])
    splits = {
        "train": {"AAAA", "CCCC"},
        "val": {"CCCC", "GGGG"},
        "test": set(),
    }

    labels = label_membership(sequences, splits)

    assert labels["in_train"].tolist() == [True, True, False, False]
    assert labels["in_val"].tolist() == [False, True, True, False]
    assert labels["in_test"].tolist() == [False, False, False, False]
    # "seen" is defined by train membership only - val/test are reported separately.
    assert labels["seen"].tolist() == [True, True, False, False]


def test_group_metrics_reports_n_mae_mse_and_spearman():
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y_pred = y_true + 0.1

    metrics = group_metrics(y_true, y_pred)

    assert metrics["n"] == 10
    assert metrics["mae"] == pytest.approx(0.1)
    assert metrics["mse"] == pytest.approx(0.01)
    assert metrics["spearman"] == pytest.approx(1.0)


def test_group_metrics_omits_spearman_when_n_too_small():
    y_true = np.arange(MIN_N_FOR_SPEARMAN - 1, dtype=np.float64)

    metrics = group_metrics(y_true, y_true + 0.5)

    assert metrics["n"] == MIN_N_FOR_SPEARMAN - 1
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["spearman"] is None


def test_compare_abs_errors_reports_mae_difference_with_bootstrap_ci_and_mann_whitney():
    rng = np.random.default_rng(0)
    err_seen = rng.uniform(0.0, 0.1, size=200)
    err_unseen = rng.uniform(0.3, 0.4, size=20)

    result = compare_abs_errors(err_seen, err_unseen, n_resamples=500, seed=0)

    # delta = MAE(unseen) - MAE(seen): positive means unseen is worse.
    assert result["mae_delta_unseen_minus_seen"] == pytest.approx(
        err_unseen.mean() - err_seen.mean()
    )
    lower, upper = result["mae_delta_bootstrap_ci95"]
    assert 0.0 < lower <= result["mae_delta_unseen_minus_seen"] <= upper
    assert result["mann_whitney_p"] < 1e-6


def test_compare_abs_errors_is_deterministic_for_a_seed():
    err_seen = np.linspace(0, 0.2, 50)
    err_unseen = np.array([0.05, 0.15, 0.3])

    a = compare_abs_errors(err_seen, err_unseen, n_resamples=200, seed=7)
    b = compare_abs_errors(err_seen, err_unseen, n_resamples=200, seed=7)

    assert a == b


def test_case_study_membership_tags_each_case_and_counts_by_type():
    seen_by_sample = {10: True, 11: False, 12: True}
    case_studies = [
        {
            "case_id": "DISCORDANT_P01",
            "sample_id": 10,
            "case_type": "Primary Discordant",
        },
        {
            "case_id": "DISCORDANT_P02",
            "sample_id": 11,
            "case_type": "Primary Discordant",
        },
        {"case_id": "CONCORDANT_C01", "sample_id": 12, "case_type": "Concordant"},
    ]

    result = case_study_membership(case_studies, seen_by_sample)

    assert [c["seen"] for c in result["cases"]] == [True, False, True]
    assert result["counts_by_type"] == {
        "Primary Discordant": {"seen": 1, "unseen": 1},
        "Concordant": {"seen": 1, "unseen": 0},
    }
    assert result["totals"] == {"seen": 2, "unseen": 1}


def _toy_inputs():
    # 6 rows, 5 unique sequences (row 5 duplicates row 0). Two sequences
    # ("GGGG", "TTTT") are absent from the old train split.
    metadata = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3, 4, 5],
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT", "ACGT", "AAAA"],
            "true_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1],
        }
    )
    predictions = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3, 4, 5],
            "pred_raw": [0.1, 0.2, 0.6, 0.8, 0.5, 0.15],
        }
    )
    splits = {"train": {"AAAA", "CCCC", "ACGT"}, "val": {"GGGG"}, "test": {"TTTT"}}
    case_studies = [
        {"case_id": "CONCORDANT_C01", "sample_id": 0, "case_type": "Concordant"},
        {
            "case_id": "DISCORDANT_R01",
            "sample_id": 3,
            "case_type": "Reverse Primary Discordant",
        },
    ]
    return metadata, predictions, splits, case_studies


def test_build_report_counts_unique_sequences_and_rows_per_group():
    metadata, predictions, splits, case_studies = _toy_inputs()

    report = build_report(metadata, predictions, splits, case_studies, n_resamples=100)

    membership = report["membership"]
    assert membership["unique_sequences"] == {"total": 5, "seen": 3, "unseen": 2}
    assert membership["rows"] == {"total": 6, "seen": 4, "unseen": 2}
    assert membership["unique_sequences_in_old_split"] == {
        "train": 3,
        "val": 1,
        "test": 1,
    }
    assert report["model_b"]["seen"]["n"] == 4
    assert report["model_b"]["unseen"]["n"] == 2
    assert report["model_b"]["unseen"]["mae"] == pytest.approx((0.3 + 0.4) / 2)
    assert "low statistical power" in report["model_b"]["note"]
    assert report["case_studies"]["totals"] == {"seen": 1, "unseen": 1}
    # The whole report must be JSON-serialisable as-is.
    json.dumps(report)


def test_build_report_rejects_misaligned_predictions():
    metadata, predictions, splits, case_studies = _toy_inputs()
    predictions = predictions.iloc[:-1]

    with pytest.raises(ValueError):
        build_report(metadata, predictions, splits, case_studies, n_resamples=10)
