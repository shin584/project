import os

import numpy as np
import xgboost as xgb
from shap_analysis import (
    PHYSICAL_DIM,
    PHYSICAL_FEATURE_NAMES,
    aggregate_physical_contribution_ratio,
    compute_tree_shap,
    dependence_plot_data,
    physical_feature_shap_summary,
    run_shap_analysis,
    save_dependence_plots,
    split_shap_by_group,
    top_k_embedding_features,
)

# Real EMBEDDING_DIM (8960) makes fitting/predicting slow for no benefit in
# these tests - exercise the same logic against a synthetic, much smaller
# embedding width, same convention as test_ablation.py's TEST_EMBEDDING_DIM.
TEST_EMBEDDING_DIM = 12


def _make_synthetic_model_and_data(n=60, seed=0, embedding_dim=TEST_EMBEDDING_DIM):
    rng = np.random.default_rng(seed)
    embedding = rng.normal(size=(n, embedding_dim)).astype(np.float32)
    physical = rng.normal(size=(n, PHYSICAL_DIM)).astype(np.float32)
    X = np.hstack([embedding, physical])
    # y leans on one embedding column and one physical column so neither
    # group's SHAP contribution is ~zero.
    y = (
        0.7 * embedding[:, 0] + 0.5 * physical[:, 0] + 0.05 * rng.normal(size=n)
    ).astype(np.float32)

    model = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, X


def test_compute_tree_shap_satisfies_additivity_against_raw_prediction():
    model, X = _make_synthetic_model_and_data()
    dmatrix = xgb.DMatrix(X)

    shap_values = compute_tree_shap(model, X)
    raw_preds = model.get_booster().predict(dmatrix, output_margin=True)
    # The Tree SHAP additivity property includes the expected-value bias term
    # (pred_contribs's dropped trailing column) - re-fetch it directly here
    # rather than have compute_tree_shap expose its own internal slicing.
    bias = model.get_booster().predict(dmatrix, pred_contribs=True)[:, -1]

    assert shap_values.shape == X.shape
    np.testing.assert_allclose(
        shap_values.sum(axis=1) + bias, raw_preds, rtol=1e-4, atol=1e-4
    )


def test_split_shap_by_group_slices_embedding_and_physical_columns():
    model, X = _make_synthetic_model_and_data()
    shap_values = compute_tree_shap(model, X)

    groups = split_shap_by_group(
        shap_values, embedding_dim=TEST_EMBEDDING_DIM, physical_dim=PHYSICAL_DIM
    )

    assert groups["embedding"].shape == (X.shape[0], TEST_EMBEDDING_DIM)
    assert groups["physical"].shape == (X.shape[0], PHYSICAL_DIM)
    np.testing.assert_array_equal(
        groups["embedding"], shap_values[:, :TEST_EMBEDDING_DIM]
    )
    np.testing.assert_array_equal(
        groups["physical"], shap_values[:, TEST_EMBEDDING_DIM:]
    )


def test_top_k_embedding_features_ranks_by_mean_abs_shap_descending():
    model, X = _make_synthetic_model_and_data()
    shap_values = compute_tree_shap(model, X)
    groups = split_shap_by_group(
        shap_values, embedding_dim=TEST_EMBEDDING_DIM, physical_dim=PHYSICAL_DIM
    )

    top = top_k_embedding_features(groups["embedding"], k=5)

    assert len(top) == 5
    mean_abs = np.abs(groups["embedding"]).mean(axis=0)
    scores = [entry["mean_abs_shap"] for entry in top]
    assert scores == sorted(scores, reverse=True)
    # Column 0 dominates the synthetic label, so it should rank first.
    assert top[0]["feature_index"] == int(np.argmax(mean_abs))


def test_top_k_embedding_features_caps_at_available_columns():
    model, X = _make_synthetic_model_and_data()
    shap_values = compute_tree_shap(model, X)
    groups = split_shap_by_group(
        shap_values, embedding_dim=TEST_EMBEDDING_DIM, physical_dim=PHYSICAL_DIM
    )

    top = top_k_embedding_features(groups["embedding"], k=20)

    assert len(top) == TEST_EMBEDDING_DIM


def test_physical_feature_shap_summary_reports_all_four_named_features():
    model, X = _make_synthetic_model_and_data()
    shap_values = compute_tree_shap(model, X)
    groups = split_shap_by_group(
        shap_values, embedding_dim=TEST_EMBEDDING_DIM, physical_dim=PHYSICAL_DIM
    )

    summary = physical_feature_shap_summary(groups["physical"])

    assert [entry["feature_name"] for entry in summary] == list(PHYSICAL_FEATURE_NAMES)
    for entry in summary:
        assert entry["mean_abs_shap"] >= 0.0


def test_aggregate_physical_contribution_ratio_is_a_valid_proportion():
    model, X = _make_synthetic_model_and_data()
    shap_values = compute_tree_shap(model, X)

    ratio = aggregate_physical_contribution_ratio(
        shap_values, embedding_dim=TEST_EMBEDDING_DIM
    )

    assert 0.0 <= ratio <= 1.0


def test_aggregate_physical_contribution_ratio_is_near_zero_when_physical_is_pure_noise():
    rng = np.random.default_rng(1)
    n = 60
    embedding = rng.normal(size=(n, TEST_EMBEDDING_DIM)).astype(np.float32)
    physical = rng.normal(size=(n, PHYSICAL_DIM)).astype(np.float32) * 1e-6
    X = np.hstack([embedding, physical])
    y = (0.7 * embedding[:, 0] + 0.05 * rng.normal(size=n)).astype(np.float32)

    model = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X, y)
    shap_values = compute_tree_shap(model, X)

    ratio = aggregate_physical_contribution_ratio(
        shap_values, embedding_dim=TEST_EMBEDDING_DIM
    )

    assert ratio < 0.05


def test_dependence_plot_data_pairs_feature_value_with_its_own_shap_column():
    model, X = _make_synthetic_model_and_data()
    shap_values = compute_tree_shap(model, X)
    groups = split_shap_by_group(
        shap_values, embedding_dim=TEST_EMBEDDING_DIM, physical_dim=PHYSICAL_DIM
    )
    X_physical = X[:, TEST_EMBEDDING_DIM:]

    data = dependence_plot_data(X_physical, groups["physical"])

    assert set(data.keys()) == set(PHYSICAL_FEATURE_NAMES)
    for i, name in enumerate(PHYSICAL_FEATURE_NAMES):
        np.testing.assert_array_equal(data[name]["feature_value"], X_physical[:, i])
        np.testing.assert_array_equal(
            data[name]["shap_value"], groups["physical"][:, i]
        )


def test_save_dependence_plots_writes_one_png_per_physical_feature(tmp_path):
    model, X = _make_synthetic_model_and_data()
    shap_values = compute_tree_shap(model, X)
    groups = split_shap_by_group(
        shap_values, embedding_dim=TEST_EMBEDDING_DIM, physical_dim=PHYSICAL_DIM
    )
    X_physical = X[:, TEST_EMBEDDING_DIM:]
    output_dir = str(tmp_path / "dependence_plots")

    paths = save_dependence_plots(X_physical, groups["physical"], output_dir)

    assert set(paths.keys()) == set(PHYSICAL_FEATURE_NAMES)
    for path in paths.values():
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


def test_run_shap_analysis_returns_the_full_acceptance_criteria_bundle():
    model, X = _make_synthetic_model_and_data()

    result = run_shap_analysis(
        model, X, embedding_dim=TEST_EMBEDDING_DIM, physical_dim=PHYSICAL_DIM, top_k=5
    )

    assert len(result["top_embedding_features"]) == 5
    assert [e["feature_name"] for e in result["physical_feature_shap"]] == list(
        PHYSICAL_FEATURE_NAMES
    )
    assert 0.0 <= result["physical_contribution_ratio"] <= 1.0
    assert result["shap_values"].shape == X.shape
    # No dependence_plot_dir given, so plotting (the bundle's only I/O step)
    # is skipped rather than writing files this test doesn't ask for.
    assert result["dependence_plot_paths"] == {}


def test_run_shap_analysis_renders_dependence_plots_when_a_dir_is_given(tmp_path):
    model, X = _make_synthetic_model_and_data()
    output_dir = str(tmp_path / "dependence_plots")

    result = run_shap_analysis(
        model,
        X,
        embedding_dim=TEST_EMBEDDING_DIM,
        physical_dim=PHYSICAL_DIM,
        dependence_plot_dir=output_dir,
    )

    assert set(result["dependence_plot_paths"].keys()) == set(PHYSICAL_FEATURE_NAMES)
    for path in result["dependence_plot_paths"].values():
        assert os.path.isfile(path)
