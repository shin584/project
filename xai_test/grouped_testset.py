"""Export the grouped Testset as test_metadata.csv-style metadata (issue #35).

Takes the `test` rows of the sequence-grouped split (#34) - every replicate
row of each test sequence, in source row order - and writes them with the
`test_metadata.csv` columns plus `cell_line_block`:

- `sample_id` is a fresh 0..N-1 Testset-internal index in row order, distinct
  from `original_id` (the row's index in `SaCas9_v4.xlsx`).
- `sequence_hash` is the per-row SHA-256 of `sequence`, as in
  `test_dataset_verifier.py`; the whole-column checksum is built the same way
  that verifier builds it, and is recorded in `test_checksum.json` together
  with the SHA-256 of the written CSV bytes.

Leaves the existing `test_metadata.csv` untouched; writes only into
`grouped_split/` (`test_metadata.csv`, `test_checksum.json`).

Usage: python grouped_testset.py [--seed 42] [--output-dir grouped_split]
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from grouped_split import OUTPUT_DIR, build_grouped_split, load_full_dataset
from test_dataset_verifier import generate_sha256

TESTSET_COLUMNS = [
    "sample_id",
    "original_id",
    "sequence",
    "true_score",
    "sequence_hash",
    "cell_line_block",
]
TESTSET_FILENAME = "test_metadata.csv"
CHECKSUM_FILENAME = "test_checksum.json"


def column_checksum(sequences: pd.Series) -> str:
    """Whole-column SHA-256, built as in `isolate_and_verify_testset`.

    `hash_pandas_object` includes the index, so pass the column with its
    0..N-1 index (as read from or written to the CSV).
    """
    return hashlib.sha256(pd.util.hash_pandas_object(sequences).values).hexdigest()


def build_grouped_testset(
    full_df: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    """The `test` rows of `full_df` per `assignments`, in `TESTSET_COLUMNS` form."""
    full_df = full_df.reset_index(drop=True)
    assignments = assignments.reset_index(drop=True)
    if not full_df["original_id"].equals(assignments["original_id"]):
        raise ValueError("assignments do not line up row-for-row with the dataset")

    is_test = (assignments["split"] == "test").to_numpy()
    testset = full_df.loc[is_test, ["original_id", "sequence", "true_score"]]
    testset["cell_line_block"] = assignments.loc[is_test, "cell_line_block"]
    testset = testset.reset_index(drop=True)

    testset["sample_id"] = range(len(testset))
    testset["sequence_hash"] = testset["sequence"].apply(generate_sha256)
    return testset[TESTSET_COLUMNS]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    full_df = load_full_dataset()
    assignments = build_grouped_split(full_df, seed=args.seed)
    testset = build_grouped_testset(full_df, assignments)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / TESTSET_FILENAME
    # Fixed line ending so `file_sha256` matches across Windows and Colab.
    testset.to_csv(csv_path, index=False, lineterminator="\n")

    record = {
        "seed": args.seed,
        "rows": len(testset),
        "sequences": int(testset["sequence"].nunique()),
        "dataset_sha256": column_checksum(testset["sequence"]),
        "file_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    with open(args.output_dir / CHECKSUM_FILENAME, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"Grouped Testset: {record['rows']} rows, {record['sequences']} sequences")
    print(f"Dataset SHA256 Checksum: {record['dataset_sha256']}")
    print(f"Wrote {csv_path} and {args.output_dir / CHECKSUM_FILENAME}")
    return record


if __name__ == "__main__":
    main()
