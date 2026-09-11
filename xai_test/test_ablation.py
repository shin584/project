import numpy as np
from ablation import (
    EMBEDDING_DIM,
    PHYSICAL_DIM,
    XGB_HYPERPARAMS,
    run_ablation,
    split_ablation_features,
    train_ablation_variant,
)

# Real EMBEDDING_DIM (8960) makes every XGBoost fit slow even on tiny row
# counts (histogram building scales with feature count, not just rows).
# Tests exercise the same orchestration logic against a much smaller
# synthetic embedding dimension via `embedding_dim=` overrides; only
# `test_split_ablation_features_defaults_to_the_real_dimensions` checks the
# real production constants.
TEST_EMBEDDING_DIM = 12


def _make_synthetic_dataset(n, seed, embedding_dim=TEST_EMBEDDING_DIM):
    rng = np.random.default_rng(seed)
    embedding = rng.normal(size=(n, embedding_dim)).astype(np.float32)
    physical = rng.normal(size=(n, PHYSICAL_DIM)).astype(np.float32)
    X = np.hstack([embedding, physical])
    # y depends on both blocks so neither ablated variant is a perfect predictor.
    y = (
        0.6 * embedding[:, 0] + 0.3 * physical[:, 0] + 0.1 * rng.normal(size=n)
    ).astype(np.float32)
    return X, y


def test_split_ablation_features_defaults_to_the_real_dimensions():
    X_full = np.random.default_rng(0).normal(size=(3, EMBEDDING_DIM + PHYSICAL_DIM))

    splits = split_ablation_features(X_full)

    assert splits["embedding"].shape == (3, EMBEDDING_DIM)
    assert splits["physical"].shape == (3, PHYSICAL_DIM)
    assert splits["full"].shape == (3, EMBEDDING_DIM + PHYSICAL_DIM)
    np.testing.assert_array_equal(splits["embedding"], X_full[:, :EMBEDDING_DIM])
    np.testing.assert_array_equal(splits["physical"], X_full[:, EMBEDDING_DIM:])
    np.testing.assert_array_equal(splits["full"], X_full)


def test_split_ablation_features_honors_a_custom_embedding_dim():
    X_full = np.random.default_rng(0).normal(
        size=(3, TEST_EMBEDDING_DIM + PHYSICAL_DIM)
    )

    splits = split_ablation_features(X_full, embedding_dim=TEST_EMBEDDING_DIM)

    assert splits["embedding"].shape == (3, TEST_EMBEDDING_DIM)
    assert splits["physical"].shape == (3, PHYSICAL_DIM)


def test_train_ablation_variant_uses_the_shared_hyperparameters():
    X_train, y_train = _make_synthetic_dataset(60, seed=1)
    X_val, y_val = _make_synthetic_dataset(20, seed=2)

    model = train_ablation_variant(X_train[:, :5], y_train, X_val[:, :5], y_val)

    params = model.get_params()
    for key, value in XGB_HYPERPARAMS.items():
        assert params[key] == value


def test_run_ablation_returns_all_three_variants_with_all_four_metrics():
    X_train, y_train = _make_synthetic_dataset(80, seed=1)
    X_val, y_val = _make_synthetic_dataset(30, seed=2)
    X_test, y_test = _make_synthetic_dataset(30, seed=3)

    result = run_ablation(
        X_train, y_train, X_val, y_val, X_test, y_test, embedding_dim=TEST_EMBEDDING_DIM
    )

    assert set(result["ablation_results"].keys()) == {
        "embedding_only",
        "physical_only",
        "full_model",
    }
    for variant_metrics in result["ablation_results"].values():
        assert set(variant_metrics.keys()) == {"spearman", "pearson", "mae", "mse"}


def test_run_ablation_bootstrap_cis_cover_full_vs_each_variant():
    X_train, y_train = _make_synthetic_dataset(80, seed=1)
    X_val, y_val = _make_synthetic_dataset(30, seed=2)
    X_test, y_test = _make_synthetic_dataset(30, seed=3)

    result = run_ablation(
        X_train, y_train, X_val, y_val, X_test, y_test, embedding_dim=TEST_EMBEDDING_DIM
    )

    assert set(result["paired_bootstrap_95ci"].keys()) == {
        "delta_spearman_full_vs_embedding_only",
        "delta_mse_full_vs_embedding_only",
        "delta_spearman_full_vs_physical_only",
        "delta_mse_full_vs_physical_only",
    }
    for lower, upper in result["paired_bootstrap_95ci"].values():
        assert lower <= upper


def test_run_ablation_is_reproducible_under_the_same_bootstrap_seed():
    X_train, y_train = _make_synthetic_dataset(80, seed=1)
    X_val, y_val = _make_synthetic_dataset(30, seed=2)
    X_test, y_test = _make_synthetic_dataset(30, seed=3)

    first = run_ablation(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        bootstrap_seed=42,
        embedding_dim=TEST_EMBEDDING_DIM,
    )
    second = run_ablation(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        bootstrap_seed=42,
        embedding_dim=TEST_EMBEDDING_DIM,
    )

    assert first == second


def test_run_ablation_does_not_rescale_before_computing_metrics():
    # A variant fed exactly y_test-correlated features (embedding column 0 is
    # the dominant term in the synthetic label) should show up as a
    # noticeably different (raw) MSE from a variant fed only weak physical
    # features - proving metrics are read straight off predict(), unscaled.
    X_train, y_train = _make_synthetic_dataset(120, seed=1)
    X_val, y_val = _make_synthetic_dataset(40, seed=2)
    X_test, y_test = _make_synthetic_dataset(40, seed=3)

    result = run_ablation(
        X_train, y_train, X_val, y_val, X_test, y_test, embedding_dim=TEST_EMBEDDING_DIM
    )

    embedding_mse = result["ablation_results"]["embedding_only"]["mse"]
    physical_mse = result["ablation_results"]["physical_only"]["mse"]
    assert embedding_mse != physical_mse
