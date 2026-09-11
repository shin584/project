import pandas as pd
from dev_split import build_dev_split
from test_dataset_verifier import (
    compute_split_indices,
    load_dataset,
    standardize_columns,
)

FULL_DATASET_PATH = "SaCas9_v4.xlsx"
TEST_METADATA_PATH = "test_metadata.csv"


def test_compute_split_indices_reproduces_the_existing_isolated_testset():
    # Regression check: the shared split logic must still produce exactly the
    # original_id set already baked into (and SHA256-verified in) test_metadata.csv.
    full_df = standardize_columns(load_dataset(FULL_DATASET_PATH))
    _, test_indices = compute_split_indices(len(full_df), seed=42)

    existing_testset = pd.read_csv(TEST_METADATA_PATH)

    assert sorted(test_indices) == sorted(existing_testset["original_id"].tolist())


def test_build_dev_split_is_disjoint_from_the_isolated_testset():
    train_df, val_df = build_dev_split(FULL_DATASET_PATH)
    testset = pd.read_csv(TEST_METADATA_PATH)

    testset_original_ids = set(testset["original_id"])
    train_original_ids = set(train_df["original_id"])
    val_original_ids = set(val_df["original_id"])

    assert train_original_ids.isdisjoint(testset_original_ids)
    assert val_original_ids.isdisjoint(testset_original_ids)
    assert train_original_ids.isdisjoint(val_original_ids)


def test_build_dev_split_sizes_match_the_10pct_convention():
    full_df = standardize_columns(load_dataset(FULL_DATASET_PATH))
    testset = pd.read_csv(TEST_METADATA_PATH)

    train_df, val_df = build_dev_split(FULL_DATASET_PATH)

    assert len(val_df) == len(testset)
    assert len(train_df) + len(val_df) + len(testset) == len(full_df)


def test_build_dev_split_has_required_columns():
    train_df, val_df = build_dev_split(FULL_DATASET_PATH)

    for df in (train_df, val_df):
        assert "sequence" in df.columns
        assert "true_score" in df.columns
        assert "original_id" in df.columns


def test_build_dev_split_is_reproducible_under_the_same_seed():
    train_a, val_a = build_dev_split(FULL_DATASET_PATH, seed=42)
    train_b, val_b = build_dev_split(FULL_DATASET_PATH, seed=42)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(val_a, val_b)


def test_build_dev_split_different_seed_produces_a_different_split():
    train_42, _ = build_dev_split(FULL_DATASET_PATH, seed=42)
    train_7, _ = build_dev_split(FULL_DATASET_PATH, seed=7)

    assert list(train_42["original_id"]) != list(train_7["original_id"])
