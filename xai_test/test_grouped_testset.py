import hashlib

import pandas as pd
import pytest
from grouped_split import DATASET_PATH, build_grouped_split, load_full_dataset
from grouped_testset import (
    TESTSET_COLUMNS,
    build_grouped_testset,
    column_checksum,
    main,
)
from test_dataset_verifier import isolate_and_verify_testset


def _toy_full_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence": ["AAA", "CCC", "GGG", "AAA", "TTT", "ccc "],
            "true_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "original_id": [10, 11, 12, 13, 14, 15],
            "Target gene": ["g"] * 6,
        }
    )


def _toy_assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "original_id": [10, 11, 12, 13, 14, 15],
            "normalized_sequence": ["AAA", "CCC", "GGG", "AAA", "TTT", "CCC"],
            "cell_line_block": ["L1", "L1", "L1", "L2", "L2", "L2"],
            "split": ["test", "test", "train", "test", "val", "test"],
        }
    )


@pytest.fixture(scope="module")
def full_df() -> pd.DataFrame:
    return load_full_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def assignments(full_df) -> pd.DataFrame:
    return build_grouped_split(full_df, seed=42)


@pytest.fixture(scope="module")
def testset(full_df, assignments) -> pd.DataFrame:
    return build_grouped_testset(full_df, assignments)


# --- unit: toy data ---


def test_testset_columns_are_test_metadata_columns_plus_cell_line_block():
    old_columns = pd.read_csv("test_metadata.csv", nrows=0).columns.tolist()

    assert TESTSET_COLUMNS == old_columns + ["cell_line_block"]


def test_build_grouped_testset_keeps_test_rows_in_row_order_with_fresh_sample_id():
    testset = build_grouped_testset(_toy_full_df(), _toy_assignments())

    assert list(testset.columns) == TESTSET_COLUMNS
    assert testset["original_id"].tolist() == [10, 11, 13, 15]
    assert testset["sample_id"].tolist() == [0, 1, 2, 3]
    assert testset["cell_line_block"].tolist() == ["L1", "L1", "L2", "L2"]
    assert testset["true_score"].tolist() == [0.1, 0.2, 0.4, 0.6]
    # `sequence` is the source row's sequence, as in test_metadata.csv.
    assert testset["sequence"].tolist() == ["AAA", "CCC", "AAA", "ccc "]


def test_build_grouped_testset_hashes_each_sequence_with_sha256():
    testset = build_grouped_testset(_toy_full_df(), _toy_assignments())

    expected = [
        hashlib.sha256(s.encode("utf-8")).hexdigest() for s in testset["sequence"]
    ]
    assert testset["sequence_hash"].tolist() == expected


def test_build_grouped_testset_rejects_mismatched_assignments():
    assignments = _toy_assignments()
    assignments.loc[0, "original_id"] = 99

    with pytest.raises(ValueError):
        build_grouped_testset(_toy_full_df(), assignments)


def test_column_checksum_matches_the_existing_verifier(tmp_path):
    _, verifier_hash = isolate_and_verify_testset(
        str(DATASET_PATH), output_csv_path=str(tmp_path / "old.csv")
    )
    old_testset = pd.read_csv("test_metadata.csv")

    assert column_checksum(old_testset["sequence"]) == verifier_hash


def test_column_checksum_depends_on_order_and_content():
    base = pd.Series(["AAA", "CCC"])

    assert column_checksum(base) == column_checksum(pd.Series(["AAA", "CCC"]))
    assert column_checksum(base) != column_checksum(pd.Series(["CCC", "AAA"]))
    assert column_checksum(base) != column_checksum(pd.Series(["AAA", "CCG"]))


# --- integration: the real SaCas9_v4.xlsx ---


def test_real_testset_is_exactly_every_row_of_the_grouped_test_split(
    assignments, testset
):
    test_rows = assignments[assignments["split"] == "test"]

    assert len(testset) == len(test_rows) == 514
    assert set(testset["original_id"]) == set(test_rows["original_id"])
    assert testset["sequence"].str.strip().str.upper().nunique() == 180
    assert testset["original_id"].is_monotonic_increasing
    assert testset["sample_id"].tolist() == list(range(514))


def test_real_testset_sequence_hashes_and_checksum(testset):
    expected = testset["sequence"].map(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    )

    assert testset["sequence_hash"].tolist() == expected.tolist()
    assert (
        column_checksum(testset["sequence"])
        == hashlib.sha256(
            pd.util.hash_pandas_object(testset["sequence"]).values
        ).hexdigest()
    )


def test_main_writes_byte_identical_files_on_rerun(tmp_path):
    first = main(["--seed", "42", "--output-dir", str(tmp_path / "a")])
    second = main(["--seed", "42", "--output-dir", str(tmp_path / "b")])

    for name in ("test_metadata.csv", "test_checksum.json"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()
    assert first == second

    written = pd.read_csv(tmp_path / "a" / "test_metadata.csv")
    assert list(written.columns) == TESTSET_COLUMNS
    assert column_checksum(written["sequence"]) == first["dataset_sha256"]
    file_bytes = (tmp_path / "a" / "test_metadata.csv").read_bytes()
    assert first["file_sha256"] == hashlib.sha256(file_bytes).hexdigest()
    assert first["seed"] == 42 and first["rows"] == 514 and first["sequences"] == 180
