"""Rebuild 9-Hydra v1's split membership and overlap categories (issue #43).

9-Hydra v1 was trained with a row-level
`train_test_split(test_size=0.1, random_state=42)` over the all-9-heads subset
of `dataset/kim_2020_table_s8.xlsx`, followed by Keras `validation_split=0.2`.
Keras takes the **last** 20% of the data it is handed, before the per-epoch
shuffle, so the validation rows are the tail of the already-shuffled 90%.
Rebuilding both steps gives back exactly which rows v1 trained on, validated
on and reported as its test set: 21,097 / 5,275 / 2,931.

Each test row is then sorted into one of the three mutually exclusive
categories defined in `CONTEXT.md`, judged against the **train role only**:

- `exact_duplicate` - this whole 30-mer appears in a train row (memorization),
- `pam_variant_overlap` - the protospacer appears in train but this 30-mer
  does not (a new PAM and 3' flank on a known target; a legitimate
  generalization test),
- `unseen_target` - the protospacer appears in no train row.

Over v1's test rows that split is 2,191 / 699 / 41.

Read-only: this module reads the spreadsheet and writes nothing. Running it
prints the membership summary to stdout.

Usage: python -m SpCas9.spcas9_v1_membership
"""

import json
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from SpCas9.spcas9_data import (
    PAM_SLICE,
    PROTOSPACER_SLICE,
    SEQUENCE_COLUMN,
    USE_COLUMN,
    filter_all_heads,
    is_published_test,
    load_spcas9_table,
    pams,
    protospacers,
)

TRAIN, VAL, TEST = "train", "val", "test"
ROLES = (TRAIN, VAL, TEST)

ROLE_COLUMN = "role"
CATEGORY_COLUMN = "category"

UNSEEN_TARGET = "unseen_target"
PAM_VARIANT_OVERLAP = "pam_variant_overlap"
EXACT_DUPLICATE = "exact_duplicate"
CATEGORIES = (UNSEEN_TARGET, PAM_VARIANT_OVERLAP, EXACT_DUPLICATE)

# v1's hyperparameters, as the notebook ran them.
TEST_SIZE = 0.1
RANDOM_STATE = 42
VALIDATION_SPLIT = 0.2

MEMBERSHIP_COLUMNS = [
    "row_index",
    SEQUENCE_COLUMN,
    "protospacer",
    "pam",
    USE_COLUMN,
    ROLE_COLUMN,
    CATEGORY_COLUMN,
    "in_pam_library",
]


def rebuild_roles(
    n: int,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    validation_split: float = VALIDATION_SPLIT,
) -> dict[str, np.ndarray]:
    """v1's train / val / test row positions over an `n`-row frame.

    The returned arrays partition `range(n)`: the test positions are
    scikit-learn's, and the train and val positions are the head and tail of
    scikit-learn's train positions, in their shuffled order, the way Keras
    `validation_split` carves them.
    """
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=test_size, random_state=random_state
    )
    split_at = int(len(train_idx) * (1.0 - validation_split))
    return {
        TRAIN: train_idx[:split_at],
        VAL: train_idx[split_at:],
        TEST: test_idx,
    }


def _blank_labels(n: int) -> np.ndarray:
    """An `n`-long array of empty strings, to be filled in by row position."""
    return np.full(n, "", dtype=object)


def role_labels(roles: dict[str, np.ndarray]) -> np.ndarray:
    """Per-row role as an array of `train` / `val` / `test` strings."""
    labels = _blank_labels(sum(len(positions) for positions in roles.values()))
    for role, positions in roles.items():
        labels[positions] = role
    return labels


def categorize(sequences: Iterable[str], train_sequences: Iterable[str]) -> list[str]:
    """Sort each sequence into one of the three overlap categories.

    `train_sequences` are the rows the model trained on; the comparison is by
    whole 30-mer first, then by protospacer.
    """
    train_sequences = list(train_sequences)
    train_30mers = set(train_sequences)
    train_protospacers = {seq[PROTOSPACER_SLICE] for seq in train_sequences}

    categories = []
    for sequence in sequences:
        if sequence in train_30mers:
            categories.append(EXACT_DUPLICATE)
        elif sequence[PROTOSPACER_SLICE] in train_protospacers:
            categories.append(PAM_VARIANT_OVERLAP)
        else:
            categories.append(UNSEEN_TARGET)
    return categories


def pam_library_protospacers(sequences: Iterable[str]) -> set[str]:
    """Protospacers that appear with more than one distinct PAM.

    The dataset holds a PAM library: 30 protospacers repeated across up to 268
    rows each that vary only at positions 24-29. Every other repeated
    protospacer in the frame repeats as a byte-identical 30-mer, so "more than
    one PAM" separates the library cleanly without a frequency threshold.
    """
    frame = pd.DataFrame(
        {
            "protospacer": [s[PROTOSPACER_SLICE] for s in sequences],
            "pam": [s[PAM_SLICE] for s in sequences],
        }
    )
    distinct_pams = frame.groupby("protospacer")["pam"].nunique()
    return set(distinct_pams[distinct_pams > 1].index)


def build_membership(
    table: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    validation_split: float = VALIDATION_SPLIT,
) -> pd.DataFrame:
    """One row per frame row: its v1 role and, for test rows, its category.

    `table` is expected to be the all-9-heads frame. `category` is empty for
    train and val rows, which are not what the audit scores.
    """
    sequence_list = table[SEQUENCE_COLUMN].tolist()
    n = len(sequence_list)
    roles = rebuild_roles(
        n,
        test_size=test_size,
        random_state=random_state,
        validation_split=validation_split,
    )
    labels = role_labels(roles)

    train_sequences = [sequence_list[i] for i in roles[TRAIN]]
    test_positions = roles[TEST]
    categories = _blank_labels(n)
    categories[test_positions] = categorize(
        (sequence_list[i] for i in test_positions), train_sequences
    )

    library = pam_library_protospacers(sequence_list)
    protospacer_column = protospacers(table)

    return pd.DataFrame(
        {
            "row_index": np.arange(n),
            SEQUENCE_COLUMN: np.asarray(sequence_list, dtype=object),
            "protospacer": protospacer_column.to_numpy(),
            "pam": pams(table).to_numpy(),
            USE_COLUMN: table[USE_COLUMN].to_numpy(),
            ROLE_COLUMN: labels,
            CATEGORY_COLUMN: categories,
            "in_pam_library": protospacer_column.isin(library).to_numpy(),
        },
        columns=MEMBERSHIP_COLUMNS,
    )


def summarize(membership: pd.DataFrame) -> dict:
    """Counts the audit report cites: roles, categories and PAM-library mass."""
    published_test = is_published_test(membership)
    test_rows = membership[membership[ROLE_COLUMN] == TEST]
    library_rows = membership[membership["in_pam_library"]]

    return {
        "rows": len(membership),
        "roles": {role: int((membership[ROLE_COLUMN] == role).sum()) for role in ROLES},
        "test_categories": {
            category: int((test_rows[CATEGORY_COLUMN] == category).sum())
            for category in CATEGORIES
        },
        "published_test_rows": int(published_test.sum()),
        "published_test_by_role": {
            role: int(((membership[ROLE_COLUMN] == role) & published_test).sum())
            for role in ROLES
        },
        "pam_library": {
            "protospacers": int(library_rows["protospacer"].nunique()),
            "rows": len(library_rows),
        },
    }


def main() -> None:
    membership = build_membership(filter_all_heads(load_spcas9_table()))
    print(json.dumps(summarize(membership), indent=2))


if __name__ == "__main__":
    main()
