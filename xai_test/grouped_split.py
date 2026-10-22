"""Sequence-grouped train/val/test split with cell_line_block (issue #34).

The rows of `SaCas9_v4.xlsx` are cell-line replicates: its 5,145 rows hold
only 1,805 distinct sequences, so the row-level seed=42 split in
`test_dataset_verifier.py` puts replicates of the same sequence on both sides
of the Testset boundary. This module splits by **sequence group** instead:

- Rows are grouped by normalized sequence (strip + upper-case). With a fixed
  seed, ~10% of the sequences go to the Testset and the same number to val;
  the rest are train. Every replicate of a sequence lands in the same split.
- Each row gets a `cell_line_block` (L1/L2/L3) from its row position
  (rows 0-1534 / 1535-3339 / 3340-5144). It is metadata only - not a model
  input and not mapped to a cell-line name, since that mapping is unconfirmed.

Read-only with respect to every existing file; writes only into
`grouped_split/` (`assignments.csv`, `summary.json`).

Usage: python grouped_split.py [--seed 42]
"""

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from test_dataset_verifier import load_dataset, standardize_columns

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "SaCas9_v4.xlsx"
OUTPUT_DIR = BASE_DIR / "grouped_split"

SPLITS = ("train", "val", "test")
CELL_LINE_BLOCK_SIZES = (1535, 1805, 1805)
TEST_FRAC = 0.1
VAL_FRAC = 0.1
OUTPUT_COLUMNS = ["original_id", "normalized_sequence", "cell_line_block", "split"]


def load_full_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    return standardize_columns(load_dataset(str(path)))


def block_labels(sizes: Sequence[int] = CELL_LINE_BLOCK_SIZES) -> list[str]:
    return [f"L{i + 1}" for i in range(len(sizes))]


def normalize_sequences(sequences: pd.Series) -> pd.Series:
    return sequences.astype(str).str.strip().str.upper()


def split_sequence_groups(
    sequences: Iterable[str],
    seed: int = 42,
    test_frac: float = TEST_FRAC,
    val_frac: float = VAL_FRAC,
) -> dict[str, str]:
    """Map each distinct sequence to `train`/`val`/`test`.

    Sequences are sorted before the seeded permutation, so the result depends
    only on the set of sequences and the seed, not on row order.
    """
    unique = np.array(sorted(set(sequences)))
    n = len(unique)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    shuffled = unique[np.random.default_rng(seed).permutation(n)]
    return {
        **{seq: "test" for seq in shuffled[:n_test]},
        **{seq: "val" for seq in shuffled[n_test : n_test + n_val]},
        **{seq: "train" for seq in shuffled[n_test + n_val :]},
    }


def assign_cell_line_blocks(
    n_rows: int, sizes: Sequence[int] = CELL_LINE_BLOCK_SIZES
) -> pd.Series:
    """`L1`/`L2`/... per row position, with block lengths given by `sizes`."""
    if sum(sizes) != n_rows:
        raise ValueError(
            f"cell_line_block sizes {tuple(sizes)} sum to {sum(sizes)}, "
            f"but the dataset has {n_rows} rows"
        )
    return pd.Series(np.repeat(block_labels(sizes), sizes), name="cell_line_block")


def validate_cell_line_blocks(
    df: pd.DataFrame, sizes: Sequence[int] = CELL_LINE_BLOCK_SIZES
) -> None:
    """Block sizes must match `sizes`, and no sequence may repeat within a block."""
    counts = df["cell_line_block"].value_counts()
    actual = tuple(int(counts.get(label, 0)) for label in block_labels(sizes))
    if actual != tuple(sizes):
        raise ValueError(f"cell_line_block sizes {actual}, expected {tuple(sizes)}")

    per_block = df.groupby("cell_line_block")["normalized_sequence"]
    repeated = per_block.size() - per_block.nunique()
    if (repeated > 0).any():
        raise ValueError(
            f"Repeated sequences within cell_line_block: "
            f"{repeated[repeated > 0].to_dict()}"
        )


def build_grouped_split(
    full_df: pd.DataFrame,
    seed: int = 42,
    block_sizes: Sequence[int] = CELL_LINE_BLOCK_SIZES,
) -> pd.DataFrame:
    """Per-row assignments: `original_id`, `normalized_sequence`, `cell_line_block`, `split`."""
    full_df = full_df.reset_index(drop=True)
    assignments = pd.DataFrame(
        {
            "original_id": full_df["original_id"],
            "normalized_sequence": normalize_sequences(full_df["sequence"]),
            "cell_line_block": assign_cell_line_blocks(len(full_df), block_sizes),
        }
    )
    validate_cell_line_blocks(assignments, block_sizes)

    split_of = split_sequence_groups(assignments["normalized_sequence"], seed=seed)
    assignments["split"] = assignments["normalized_sequence"].map(split_of)
    return assignments[OUTPUT_COLUMNS]


def summarize(assignments: pd.DataFrame) -> dict:
    """Row and sequence counts per split, with rows broken down by `cell_line_block`."""
    blocks = sorted(set(block_labels()) | set(assignments["cell_line_block"]))

    def counts(part: pd.DataFrame) -> dict:
        by_block = part["cell_line_block"].value_counts()
        return {
            "rows": len(part),
            "sequences": int(part["normalized_sequence"].nunique()),
            "rows_by_cell_line_block": {b: int(by_block.get(b, 0)) for b in blocks},
        }

    summary = {s: counts(assignments[assignments["split"] == s]) for s in SPLITS}
    summary["total"] = counts(assignments)
    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    assignments = build_grouped_split(load_full_dataset(), seed=args.seed)
    summary = summarize(assignments)

    OUTPUT_DIR.mkdir(exist_ok=True)
    assignments.to_csv(OUTPUT_DIR / "assignments.csv", index=False)
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "counts": summary}, f, indent=2)

    for split in (*SPLITS, "total"):
        split_counts = summary[split]
        print(
            f"{split:>5}: {split_counts['rows']:>5} rows, "
            f"{split_counts['sequences']:>5} sequences, "
            f"rows by cell_line_block {split_counts['rows_by_cell_line_block']}"
        )
    print(f"Wrote {OUTPUT_DIR / 'assignments.csv'} and {OUTPUT_DIR / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
