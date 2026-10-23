import builtins

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from test_spcas9_data import make_sequence, numbered_protospacer

from SpCas9.spcas9_data import filter_all_heads, load_spcas9_table
from SpCas9.spcas9_v1_membership import (
    CATEGORIES,
    EXACT_DUPLICATE,
    PAM_VARIANT_OVERLAP,
    ROLES,
    UNSEEN_TARGET,
    build_membership,
    categorize,
    rebuild_roles,
    summarize,
)


def _toy_frame(sequences: list[str], use: list[str] | None = None) -> pd.DataFrame:
    """The columns `build_membership` reads off the all-9-heads frame."""
    return pd.DataFrame(
        {
            "sequence": sequences,
            "use": use if use is not None else ["Training"] * len(sequences),
        }
    )


def _numbered_sequences(n: int) -> list[str]:
    """`n` rows with distinct protospacers, numbered in base-4 over ACGT."""
    return [make_sequence(protospacer=numbered_protospacer(i)) for i in range(n)]


@pytest.fixture(scope="module")
def real_membership() -> pd.DataFrame:
    return build_membership(filter_all_heads(load_spcas9_table()))


# --- unit: role rebuild ---


def test_rebuild_roles_matches_a_direct_train_test_split_with_the_same_seed():
    roles = rebuild_roles(100, test_size=0.1, random_state=42)

    train_idx, test_idx = train_test_split(
        np.arange(100), test_size=0.1, random_state=42
    )
    assert roles["test"].tolist() == test_idx.tolist()
    assert roles["train"].tolist() == train_idx[:72].tolist()
    assert roles["val"].tolist() == train_idx[72:].tolist()


def test_rebuild_roles_takes_the_last_fifth_of_the_shuffled_train_rows_as_val():
    roles = rebuild_roles(100)

    train_idx, _ = train_test_split(np.arange(100), test_size=0.1, random_state=42)
    assert roles["val"].tolist() == train_idx[-18:].tolist()


def test_rebuild_roles_covers_every_row_exactly_once():
    n = 257
    roles = rebuild_roles(n)

    covered = np.concatenate([roles[role] for role in ROLES])
    assert sorted(covered.tolist()) == list(range(n))


def test_rebuild_roles_differs_across_seeds():
    assert (
        rebuild_roles(100, random_state=42)["test"].tolist()
        != rebuild_roles(100, random_state=7)["test"].tolist()
    )


# --- unit: overlap categories ---


def test_categorize_sorts_hand_built_sequences_into_the_three_categories():
    trained_on = [make_sequence(protospacer="G" * 20, pam="TGG")]

    categories = categorize(
        [
            make_sequence(protospacer="G" * 20, pam="TGG"),
            make_sequence(protospacer="G" * 20, pam="CGG"),
            make_sequence(protospacer="G" * 20, pam="TGG", tail="AAA"),
            make_sequence(protospacer="C" * 20, pam="TGG"),
        ],
        trained_on,
    )

    assert categories == [
        EXACT_DUPLICATE,
        PAM_VARIANT_OVERLAP,
        PAM_VARIANT_OVERLAP,
        UNSEEN_TARGET,
    ]


def test_categorize_gives_every_row_exactly_one_known_category():
    sequences = _numbered_sequences(20)

    categories = categorize(sequences, sequences[:10])

    assert len(categories) == len(sequences)
    assert set(categories) <= set(CATEGORIES)


def test_categorize_against_no_training_rows_calls_everything_unseen():
    assert categorize(_numbered_sequences(5), []) == [UNSEEN_TARGET] * 5


# --- unit: membership frame ---


def test_build_membership_labels_every_row_with_a_role_and_test_rows_only_with_a_category():
    membership = build_membership(_toy_frame(_numbered_sequences(100)))

    assert len(membership) == 100
    assert set(membership["role"]) == set(ROLES)
    test_rows = membership[membership["role"] == "test"]
    assert set(test_rows["category"]) <= set(CATEGORIES)
    assert set(membership[membership["role"] != "test"]["category"]) == {""}


def test_build_membership_keeps_the_frames_row_order_and_windows():
    sequences = [
        make_sequence(protospacer="G" * 20, pam="TGG"),
        make_sequence(protospacer="C" * 20, pam="AGG"),
        make_sequence(protospacer="T" * 20, pam="CGG"),
        make_sequence(protospacer="A" * 20, pam="GGG"),
    ]

    membership = build_membership(_toy_frame(sequences), test_size=0.25)

    assert membership["row_index"].tolist() == [0, 1, 2, 3]
    assert membership["sequence"].tolist() == sequences
    assert membership["protospacer"].tolist() == [
        "G" * 20,
        "C" * 20,
        "T" * 20,
        "A" * 20,
    ]
    assert membership["pam"].tolist() == ["TGG", "AGG", "CGG", "GGG"]
    # Every role is populated, so the category logic is not running vacuously.
    assert sorted(membership["role"]) == ["test", "train", "train", "val"]


def test_summarize_counts_roles_categories_and_published_test_rows():
    sequences = _numbered_sequences(100)
    use = ["Test" if i % 2 else "Training" for i in range(100)]

    summary = summarize(build_membership(_toy_frame(sequences, use)))

    assert summary["rows"] == 100
    assert sum(summary["roles"].values()) == 100
    assert sum(summary["test_categories"].values()) == summary["roles"]["test"]
    assert summary["published_test_rows"] == 50
    assert sum(summary["published_test_by_role"].values()) == 50


# --- regression: the real kim_2020_table_s8.xlsx ---


def test_real_membership_rebuilds_v1s_pinned_role_counts(real_membership):
    summary = summarize(real_membership)

    assert summary["rows"] == 29_303
    assert summary["roles"] == {"train": 21_097, "val": 5_275, "test": 2_931}


def test_real_membership_rebuilds_v1s_pinned_category_counts(real_membership):
    assert summarize(real_membership)["test_categories"] == {
        UNSEEN_TARGET: 2_191,
        PAM_VARIANT_OVERLAP: 699,
        EXACT_DUPLICATE: 41,
    }


def test_real_published_test_rows_split_by_v1_role(real_membership):
    summary = summarize(real_membership)

    assert summary["published_test_rows"] == 3_847
    assert summary["published_test_by_role"] == {
        "train": 2_781,
        "val": 665,
        "test": 401,
    }


def test_real_pam_library_covers_the_pinned_protospacers_and_rows(real_membership):
    assert summarize(real_membership)["pam_library"] == {
        "protospacers": 30,
        "rows": 7_197,
    }


def test_real_membership_covers_every_row_exactly_once(real_membership):
    assert real_membership["row_index"].tolist() == list(range(len(real_membership)))

    roles = real_membership["role"].value_counts()
    assert set(roles.index) == set(ROLES)
    assert roles.sum() == len(real_membership) == 29_303


def test_real_test_rows_all_carry_a_category_and_other_rows_carry_none(
    real_membership,
):
    categories = real_membership.groupby("role")["category"].value_counts()

    assert set(categories["test"].index) == set(CATEGORIES)
    assert categories["test"].sum() == 2_931
    assert categories["train"].to_dict() == {"": 21_097}
    assert categories["val"].to_dict() == {"": 5_275}


def test_rebuilding_the_real_membership_writes_no_files(monkeypatch):
    """Issue #43: "The code only reads existing files and writes nothing.\""""
    real_open = builtins.open

    def read_only_open(file, mode="r", *args, **kwargs):
        if set(mode) & set("wax+"):
            raise AssertionError(f"opened {file!r} for writing (mode {mode!r})")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only_open)

    summarize(build_membership(filter_all_heads(load_spcas9_table())))
