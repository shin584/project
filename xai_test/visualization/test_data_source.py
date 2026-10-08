"""Contract tests for the precomputed Model B data source (issue #20).

Runs over every Testset sample against the real `final_analysis_result/`
export + `test_metadata.csv`, so the dashboard can trust each bundle's shape
and the explicit absence of optional fields without re-checking them.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
from charts import prediction_detail_table
from data_source import (
    CLS_GROUP_NAME,
    DISTAL_REGION,
    HIDDEN_DIM,
    N_COMPLEX_SCENARIOS,
    N_TOKENS,
    PAM_REGION,
    PHYSICAL_FEATURE_KEYS,
    SEED_REGION,
    SEQ_LEN,
    SHAP_GROUP_COUNT,
    TOKEN_NT_SPAN,
    PrecomputedDataSource,
    token_grouped_shap,
)

N_TESTSET = 514
N_CASE_STUDIES = 15


@pytest.fixture(scope="module")
def source():
    return PrecomputedDataSource()


@pytest.fixture(scope="module")
def bundles(source):
    return [source.get_sample(i) for i in source.sample_ids()]


def test_sample_ids_cover_the_testset(source):
    assert source.sample_ids() == list(range(N_TESTSET))


@pytest.mark.parametrize("bad_id", [-1, N_TESTSET])
def test_unknown_sample_id_raises(source, bad_id):
    with pytest.raises(KeyError):
        source.get_sample(bad_id)


def test_every_bundle_has_the_expected_shapes(bundles):
    for b in bundles:
        assert len(b.sequence) == SEQ_LEN
        assert set(b.sequence) <= set("ACGT")
        assert b.ism_delta.shape == (SEQ_LEN, 3)
        assert len(b.ism_alt_bases) == SEQ_LEN
        for pos, alts in enumerate(b.ism_alt_bases):
            assert len(alts) == 3
            assert b.sequence[pos] not in alts
            assert list(alts) == sorted(alts)
        assert b.complex_mismatch_delta.shape == (N_COMPLEX_SCENARIOS,)
        assert b.attention_rollout.shape == (SEQ_LEN, SEQ_LEN)
        assert len(b.shap_groups) == SHAP_GROUP_COUNT
        assert set(b.physical_features) == set(PHYSICAL_FEATURE_KEYS)


def test_no_nan_anywhere(bundles):
    for b in bundles:
        assert np.isfinite(b.prediction)
        assert np.isfinite(b.ism_delta).all()
        assert np.isfinite(b.complex_mismatch_delta).all()
        assert np.isfinite(b.attention_rollout).all()
        assert np.isfinite(b.shap_base_value)
        assert all(np.isfinite(g.shap_value) for g in b.shap_groups)
        assert all(np.isfinite(v) for v in b.physical_features.values())
        if b.ig_attribution is not None:
            assert np.isfinite(b.ig_attribution).all()


def test_testset_truth_is_present_and_error_is_consistent(bundles):
    for b in bundles:
        assert b.true_score is not None
        assert b.error == pytest.approx(abs(b.true_score - b.prediction))


def test_ig_is_explicitly_absent_outside_the_case_studies(source, bundles):
    with_ig = [b for b in bundles if b.ig_attribution is not None]
    assert len(with_ig) == N_CASE_STUDIES
    for b in with_ig:
        assert b.ig_attribution.shape == (SEQ_LEN,)
    assert source.ig_sample_ids() == sorted(b.sample_id for b in with_ig)


def test_shap_groups_are_seven_tokens_then_four_physical_features(bundles):
    names = [g.name for g in bundles[0].shap_groups]
    assert names[0] == CLS_GROUP_NAME
    assert names[-4:] == ["MFE", "ΔG", "Tm", "GC"]
    for b in bundles:
        # Token labels show the 6-mer each token covers, in sequence order.
        for k, g in enumerate(b.shap_groups[1:7]):
            start = TOKEN_NT_SPAN * k
            assert g.feature_value == b.sequence[start : start + TOKEN_NT_SPAN]
            assert g.feature_value in g.label


def test_token_grouped_shap_is_a_signed_sum_per_token():
    rng = np.random.default_rng(0)
    row = rng.normal(size=N_TOKENS * HIDDEN_DIM + 4)
    sequence = "ACGTAC" * 6
    physical = {"mfe": -1.0, "dg": -2.0, "tm": 60.0, "gc": 50.0}

    groups = token_grouped_shap(row, sequence, physical)

    per_token = row[: N_TOKENS * HIDDEN_DIM].reshape(N_TOKENS, HIDDEN_DIM).sum(1)
    assert [g.shap_value for g in groups[:N_TOKENS]] == pytest.approx(per_token)
    assert [g.shap_value for g in groups[N_TOKENS:]] == pytest.approx(row[-4:])
    assert sum(g.shap_value for g in groups) == pytest.approx(row.sum())
    assert [g.feature_value for g in groups[N_TOKENS:]] == [
        "-1.00",
        "-2.00",
        "60.00",
        "50.00",
    ]


def test_shap_base_value_is_shared_across_samples(bundles):
    assert len({b.shap_base_value for b in bundles}) == 1


def test_complex_scenarios_have_one_label_per_bar(source):
    labels = source.complex_scenarios()
    assert len(labels) == N_COMPLEX_SCENARIOS
    assert [s.region for s in labels] == ["Seed"] * 5 + ["Distal"] * 5 + [
        "Intermittent"
    ] * 5
    assert all(s.mutation_id for s in labels)


def test_ranking_is_sorted_by_model_b_prediction(source, bundles):
    ranking = source.ranking()
    assert len(ranking) == N_TESTSET
    assert ranking["prediction"].is_monotonic_decreasing
    by_id = {b.sample_id: b.prediction for b in bundles}
    for _, row in ranking.head(10).iterrows():
        assert row["prediction"] == pytest.approx(by_id[row["sample_id"]])


def test_bundle_never_exposes_model_a_or_case_labels(bundles):
    fields = set(vars(bundles[0]))
    assert not any("model_a" in f or "case" in f for f in fields)


def test_detail_table_shows_truth_only_when_present(bundles):
    with_truth = prediction_detail_table(bundles[0])
    assert {"실측값", "오차"} <= set(with_truth.index)

    without_truth = prediction_detail_table(
        dataclasses.replace(bundles[0], true_score=None, error=None)
    )
    assert "실측값" not in without_truth.index
    assert "오차" not in without_truth.index
    assert "Model B 예측값" in without_truth.index


def test_regions_match_the_pipeline_constants():
    # The dashboard keeps its own copy (it is a standalone app); guard drift
    # against the half-open boundaries the analysis pipeline uses.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import integrated_gradients as ig
    import mismatch_profiling as mp

    assert SEED_REGION == (mp.SEED_REGION_START, mp.SEED_REGION_END - 1)
    assert DISTAL_REGION == (mp.DISTAL_REGION_START, mp.DISTAL_REGION_END - 1)
    assert PAM_REGION == (ig.PAM_REGION_START, ig.PAM_REGION_END - 1)
