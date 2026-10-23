import json

import numpy as np
import pandas as pd
import pytest
from test_spcas9_data import make_sequence

from SpCas9.spcas9_data import HEAD_NAMES, output_name
from SpCas9.spcas9_v1_audit import (
    build_report,
    full_metrics,
    holm_bonferroni,
    pred_column,
    predictions_by_head,
    true_column,
)

# Per-row absolute error baked into the toy frame, by (role, category).
ERRORS = {
    ("train", ""): 0.01,
    ("val", ""): 0.02,
    ("test", "unseen_target"): 0.20,
    ("test", "pam_variant_overlap"): 0.10,
    ("test", "exact_duplicate"): 0.05,
}
COUNTS = {
    ("train", ""): 20,
    ("val", ""): 10,
    ("test", "unseen_target"): 15,
    ("test", "pam_variant_overlap"): 12,
    ("test", "exact_duplicate"): 3,
}


def _toy_predictions(errors: dict = ERRORS, published_every: int = 2) -> pd.DataFrame:
    """A predictions frame with a known, constant error per (role, category).

    Labels rise with the row number, so every group's predictions are a
    perfect monotone (and linear) function of its labels. Every
    `published_every`-th row reads `Use = Test`.
    """
    rows = []
    for (role, category), n in COUNTS.items():
        for _ in range(n):
            i = len(rows)
            row = {
                "row_index": i,
                "sequence": make_sequence(),
                "protospacer": "A" * 20,
                "use": "Test" if i % published_every == 0 else "Training",
                "role": role,
                "category": category,
                "in_pam_library": False,
            }
            for head in HEAD_NAMES:
                label = 0.01 * i
                row[true_column(head)] = label
                row[pred_column(head)] = label + errors[(role, category)]
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def report() -> dict:
    return build_report(_toy_predictions(), n_resamples=200)


# --- report builder ---


def test_category_metrics_give_group_sizes_and_known_errors(report):
    for head in HEAD_NAMES:
        metrics = report["category_metrics"][head]
        for category, expected_mae in (
            ("unseen_target", 0.20),
            ("pam_variant_overlap", 0.10),
            ("exact_duplicate", 0.05),
        ):
            assert metrics[category]["n"] == COUNTS[("test", category)]
            assert metrics[category]["mae"] == pytest.approx(expected_mae)
            assert metrics[category]["mse"] == pytest.approx(expected_mae**2)
        for category in ("unseen_target", "pam_variant_overlap"):
            assert metrics[category]["spearman"] == pytest.approx(1.0)
            assert metrics[category]["pearson"] == pytest.approx(1.0)


def test_exact_duplicate_has_no_correlation_and_no_test(report):
    for head in HEAD_NAMES:
        assert set(report["category_metrics"][head]["exact_duplicate"]) == {
            "n",
            "mae",
            "mse",
        }
    comparison = report["unseen_target_vs_pam_variant_overlap"]
    assert comparison["groups"] == {
        "seen": "pam_variant_overlap",
        "unseen": "unseen_target",
    }
    assert "exact_duplicate" not in json.dumps(comparison["heads"])


def test_category_tests_are_signed_unseen_minus_overlap_and_holm_corrected(report):
    heads = report["unseen_target_vs_pam_variant_overlap"]["heads"]
    assert list(heads) == list(HEAD_NAMES)
    for test in heads.values():
        delta = test["mae_delta_unseen_target_minus_pam_variant_overlap"]
        assert delta == pytest.approx(0.10)
        lower, upper = test["mae_delta_bootstrap_ci95"]
        assert lower == pytest.approx(0.10) and upper == pytest.approx(0.10)
        assert test["mann_whitney_p_holm"] >= test["mann_whitney_p"]


def test_published_test_rows_are_split_by_role_and_only_test_is_comparable(report):
    frame = _toy_predictions()
    published = frame[frame["use"] == "Test"]
    by_role = report["published_test_by_role"]
    for role in ("train", "val", "test"):
        assert by_role[role]["n"] == int((published["role"] == role).sum())
        assert by_role[role]["comparable_to_kim_2020"] is (role == "test")
        assert set(by_role[role]["heads"]) == set(HEAD_NAMES)


def test_sanity_check_detects_error_rising_train_to_val_to_test(report):
    sanity = report["role_sanity_check"]
    assert sanity["rises_train_to_val_to_test"] is True
    assert sanity["heads_rising"] == 9
    assert sanity["train_below_val_and_test"] is True
    assert sanity["held_out_minus_train_mae"] == pytest.approx(
        np.mean([0.02] * 10 + [0.20] * 15 + [0.10] * 12 + [0.05] * 3) - 0.01
    )
    assert len(sanity["control_seed_held_out_minus_train_mae"]) == 10
    assert sanity["mean_mae_over_heads"]["train"] == pytest.approx(0.01)


def test_sanity_check_flags_a_split_where_val_beats_train():
    flipped = {**ERRORS, ("train", ""): 0.03}
    sanity = build_report(_toy_predictions(flipped), n_resamples=50)[
        "role_sanity_check"
    ]
    assert sanity["rises_train_to_val_to_test"] is False
    assert sanity["heads_rising"] == 0
    assert sanity["train_below_val_and_test"] is False


def test_sanity_check_separates_train_lowest_from_strictly_rising():
    # Val worse than test, as with real held-out noise: train is still lowest.
    noisy = {**ERRORS, ("val", ""): 0.30}
    sanity = build_report(_toy_predictions(noisy), n_resamples=50)["role_sanity_check"]
    assert sanity["rises_train_to_val_to_test"] is False
    assert sanity["train_below_val_and_test"] is True


def test_report_is_strict_json(report):
    json.dumps(report, allow_nan=False)


def test_membership_summary_counts_roles_and_categories(report):
    assert report["membership"]["roles"] == {"train": 20, "val": 10, "test": 30}
    assert report["membership"]["test_categories"] == {
        "unseen_target": 15,
        "pam_variant_overlap": 12,
        "exact_duplicate": 3,
    }


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_full_metrics_turns_an_undefined_correlation_into_none():
    metrics = full_metrics(np.linspace(0, 1, 12), np.full(12, 0.5))
    assert metrics["spearman"] is None
    assert metrics["pearson"] is None
    json.dumps(metrics, allow_nan=False)


# --- Holm-Bonferroni ---


def test_holm_bonferroni_matches_a_hand_computed_step_down():
    # Sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> 0.06 (monotone).
    adjusted = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_bonferroni_caps_at_one():
    assert holm_bonferroni([0.5, 0.6]) == pytest.approx([1.0, 1.0])


# --- model output plumbing (no model loaded) ---


def test_predictions_are_keyed_by_output_name_not_position():
    names = [output_name(head) for head in HEAD_NAMES]
    hf, esp = names.index("Output_SpCas9_HF_1"), names.index("Output_eSpCas9_1_1_")
    names[hf], names[esp] = names[esp], names[hf]
    outputs = [np.full((2, 1), float(i)) for i in range(len(names))]

    preds = predictions_by_head(names, outputs)

    assert preds["SpCas9-HF.1"].tolist() == [float(esp)] * 2
    assert preds["eSpCas9(1.1)"].tolist() == [float(hf)] * 2
    assert list(preds) == list(HEAD_NAMES)


def test_predictions_by_head_rejects_a_missing_head():
    names = [output_name(head) for head in HEAD_NAMES[:-1]]
    with pytest.raises(KeyError, match="evoCas9"):
        predictions_by_head(names, [np.zeros((1, 1))] * len(names))
