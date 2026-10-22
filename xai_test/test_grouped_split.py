import pandas as pd
import pytest
from grouped_split import (
    DATASET_PATH,
    assign_cell_line_blocks,
    build_grouped_split,
    load_full_dataset,
    split_sequence_groups,
    summarize,
    validate_cell_line_blocks,
)


def _toy_df(sequences: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence": sequences,
            "true_score": [0.5] * len(sequences),
            "original_id": range(len(sequences)),
        }
    )


@pytest.fixture(scope="module")
def full_df() -> pd.DataFrame:
    return load_full_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def assignments(full_df) -> pd.DataFrame:
    return build_grouped_split(full_df, seed=42)


# --- unit: sequence-group split ---


def test_split_sequence_groups_sends_each_sequence_to_exactly_one_split():
    sequences = [f"SEQ{i:03d}" for i in range(100)]

    split_of = split_sequence_groups(sequences, seed=0, test_frac=0.1, val_frac=0.1)

    assert set(split_of) == set(sequences)
    counts = pd.Series(split_of).value_counts()
    assert counts["test"] == 10
    assert counts["val"] == 10
    assert counts["train"] == 80


def test_split_sequence_groups_ignores_input_order():
    sequences = [f"SEQ{i:03d}" for i in range(50)]

    forward = split_sequence_groups(sequences, seed=3)
    backward = split_sequence_groups(list(reversed(sequences)), seed=3)

    assert forward == backward


def test_split_sequence_groups_differs_across_seeds():
    sequences = [f"SEQ{i:03d}" for i in range(100)]

    assert split_sequence_groups(sequences, seed=1) != split_sequence_groups(
        sequences, seed=2
    )


# --- unit: cell_line_block ---


def test_assign_cell_line_blocks_uses_row_position():
    blocks = assign_cell_line_blocks(6, sizes=(1, 2, 3))

    assert blocks.tolist() == ["L1", "L2", "L2", "L3", "L3", "L3"]


def test_assign_cell_line_blocks_rejects_wrong_row_count():
    with pytest.raises(ValueError):
        assign_cell_line_blocks(5, sizes=(1, 2, 3))


def test_validate_cell_line_blocks_rejects_repeated_sequence_within_a_block():
    df = pd.DataFrame(
        {
            "normalized_sequence": ["AAA", "AAA", "CCC"],
            "cell_line_block": ["L1", "L1", "L2"],
        }
    )

    with pytest.raises(ValueError):
        validate_cell_line_blocks(df, sizes=(2, 1))


# --- unit: build on toy data ---


def test_build_grouped_split_keeps_replicates_together_and_normalizes():
    # Three "cell lines" of the same 20 sequences, with case/whitespace noise.
    base = [f"ACGT{i:02d}" for i in range(20)]
    df = _toy_df(base + [s.lower() for s in base] + [f" {s} " for s in base])

    result = build_grouped_split(df, seed=0, block_sizes=(20, 20, 20))

    assert list(result.columns) == [
        "original_id",
        "normalized_sequence",
        "cell_line_block",
        "split",
    ]
    assert result["normalized_sequence"].nunique() == 20
    assert (result.groupby("normalized_sequence")["split"].nunique() == 1).all()


def test_summarize_reports_rows_and_sequences_with_block_breakdown():
    df = pd.DataFrame(
        {
            "original_id": [0, 1, 2, 3],
            "normalized_sequence": ["A", "B", "A", "B"],
            "cell_line_block": ["L1", "L1", "L2", "L2"],
            "split": ["test", "train", "test", "train"],
        }
    )

    summary = summarize(df)

    assert summary["test"] == {
        "rows": 2,
        "sequences": 1,
        "rows_by_cell_line_block": {"L1": 1, "L2": 1, "L3": 0},
    }
    assert summary["val"]["rows"] == 0
    assert summary["total"]["rows"] == 4
    assert summary["total"]["sequences"] == 2


# --- integration: the real SaCas9_v4.xlsx ---


def test_real_split_has_no_sequence_in_more_than_one_split(assignments):
    splits_per_sequence = assignments.groupby("normalized_sequence")["split"].nunique()

    assert (splits_per_sequence == 1).all()
    sequences_per_split = assignments.groupby("split")["normalized_sequence"].nunique()
    assert sequences_per_split.sum() == assignments["normalized_sequence"].nunique()


def test_real_split_assigns_every_row_exactly_once(full_df, assignments):
    assert len(assignments) == len(full_df) == 5145
    assert assignments["original_id"].is_unique
    assert set(assignments["original_id"]) == set(full_df["original_id"])
    assert set(assignments["split"]) == {"train", "val", "test"}


def test_real_testset_holds_about_10pct_of_sequences(assignments):
    summary = summarize(assignments)

    assert summary["total"]["sequences"] == 1805
    assert summary["test"]["sequences"] == 180
    # Pinned for seed=42 so a change to the split shows up as a diff here.
    assert summary["test"]["rows"] == 514
    assert sum(summary[s]["rows"] for s in ("train", "val", "test")) == 5145


def test_real_split_is_deterministic_for_a_seed(full_df, assignments):
    rerun = build_grouped_split(full_df, seed=42)

    pd.testing.assert_frame_equal(assignments, rerun)
    other = build_grouped_split(full_df, seed=7)
    assert not assignments["split"].equals(other["split"])


def test_real_cell_line_blocks_have_expected_sizes_and_no_repeats(assignments):
    assert assignments["cell_line_block"].value_counts().to_dict() == {
        "L1": 1535,
        "L2": 1805,
        "L3": 1805,
    }
    per_block = assignments.groupby("cell_line_block")["normalized_sequence"]
    assert (per_block.nunique() == per_block.size()).all()
    # Block boundaries follow row order.
    ordered = assignments.sort_values("original_id")["cell_line_block"]
    assert ordered.iloc[1534] == "L1" and ordered.iloc[1535] == "L2"
    assert ordered.iloc[3339] == "L2" and ordered.iloc[3340] == "L3"
