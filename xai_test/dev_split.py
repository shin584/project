"""Dev split (train/val) for Model B ablation retraining (Step 4-1, issue #13).

The train.csv/val.csv/test.csv checked into this project predate the current
seed=42 Testset isolation (test_dataset_verifier.py / test_metadata.csv) and
don't correspond to it: 460 of the Testset's 464 unique sequences also
appear in train.csv. Training the ablation variants on that split would mean
they'd already seen ~99% of the Testset's unique content, invalidating the
Testset evaluation this issue's paired bootstrap depends on.

Instead, `build_dev_split` reconstructs train/val fresh from the same
seed=42 permutation `test_dataset_verifier.compute_split_indices` uses to
isolate the Testset. `train_val_indices` and `test_indices` are complementary
halves of one `torch.randperm` call, so the dev split is disjoint from the
Testset by construction - no post-hoc filtering needed. Val is sized to
match the Testset (514 rows); the remainder is train.
"""

import pandas as pd
from test_dataset_verifier import (
    compute_split_indices,
    load_dataset,
    standardize_columns,
)


def build_dev_split(
    full_dataset_path: str, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns `(train_df, val_df)`, both disjoint from the seed=42-isolated Testset."""
    full_df = standardize_columns(load_dataset(full_dataset_path))
    dataset_size = len(full_df)

    train_val_indices, test_indices = compute_split_indices(dataset_size, seed)
    val_size = len(test_indices)
    train_indices = train_val_indices[:-val_size]
    val_indices = train_val_indices[-val_size:]

    train_df = full_df.iloc[train_indices].reset_index(drop=True)
    val_df = full_df.iloc[val_indices].reset_index(drop=True)
    return train_df, val_df
