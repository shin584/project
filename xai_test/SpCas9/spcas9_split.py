"""Publish the 9-Hydra split to disk with a checksum sidecar (issue #45).

One command writes the 9-Hydra split as `train.csv` / `val.csv` / `test.csv`
plus `split_checksum.json`, so every later step - training, scoring, the v1
comparison - reads the same fixed row sets instead of re-deriving them.

- **Train and test come from the spreadsheet's `Use (Training or Test)` column
  verbatim** over the at-least-one-head frame: 32,810 train / 5,186 test.
  Nothing here resamples that boundary, and a `use` value other than
  `Training` or `Test` is refused rather than guessed. See
  `docs/adr/0004-honor-kim-2020-use-column.md` for why the published column
  wins over a grouped split of our own.
- **Validation is carved only from the training rows**, grouped by
  protospacer at ~10% with `np.random.default_rng(42)` in the style of
  `grouped_split.py`, because the file ships no validation column and v1's
  Keras `validation_split=0.2` leaked protospacers into early stopping. The
  carve runs over the sorted distinct protospacers, so it depends on the set
  of protospacers and the seed, never on row order.
- Each row carries what the later steps stratify on without reopening the
  spreadsheet: its `protospacer`, whether that protospacer is one of the 30
  **PAM-library** protospacers, and the **tRNA feature** - metadata only, not
  a model input (issue #42's Out of Scope). The nine head columns ride along
  so the masked-loss build can read labels, and their NaN pattern, from the
  same file.
- `split_checksum.json` records the seed, row and protospacer counts per
  split, the same counts cut the published way under `published` (train+val
  together, the 32,810 / 5,186 figures ADR 0004 quotes), the SHA-256 of each
  written CSV, and `dataset_sha256`, the SHA-256 of the spreadsheet itself.
  That pin is what stops a re-exported spreadsheet or a stale CSV from
  silently changing what the 9-Hydra Testset means. It is also the only file
  of the set git tracks - `*.csv` is ignored repo-wide.
- Because ADR 0004 makes the published split a fixed artifact, `main` refuses
  a non-default `--seed` that would overwrite it in place; a sweep has to name
  its own `--output-dir`.

Reads the spreadsheet; writes only into `SpCas9/published_split/`.

Usage: python -m SpCas9.spcas9_split [--seed 42] [--output-dir ...]
"""

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from SpCas9.spcas9_data import (
    DATASET_PATH,
    HEAD_NAMES,
    SEQUENCE_COLUMN,
    TRNA_COLUMN,
    USE_COLUMN,
    USE_VALUES,
    filter_any_head,
    is_published_test,
    load_spcas9_table,
    pam_library_protospacers,
    protospacers,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "published_split"

TRAIN, VAL, TEST = "train", "val", "test"
SPLITS = (TRAIN, VAL, TEST)

SPLIT_COLUMN = "split"
PROTOSPACER_COLUMN = "protospacer"
PAM_LIBRARY_COLUMN = "in_pam_library"
ROW_INDEX_COLUMN = "row_index"

SEED = 42
VAL_FRAC = 0.1

CHECKSUM_FILENAME = "split_checksum.json"
SPLIT_FILENAMES = {split: f"{split}.csv" for split in SPLITS}

OUTPUT_COLUMNS = [
    ROW_INDEX_COLUMN,
    SEQUENCE_COLUMN,
    PROTOSPACER_COLUMN,
    PAM_LIBRARY_COLUMN,
    TRNA_COLUMN,
    USE_COLUMN,
    SPLIT_COLUMN,
    *HEAD_NAMES,
]


def carve_validation_protospacers(
    training_protospacers: Iterable[str],
    seed: int = SEED,
    val_frac: float = VAL_FRAC,
) -> set[str]:
    """The protospacers held out for validation, grouped and seeded.

    Sorts the distinct protospacers before the seeded permutation, so the
    result depends only on the set of protospacers and the seed - not on the
    order the rows happen to arrive in.
    """
    unique = np.array(sorted(set(training_protospacers)))
    n_val = int(len(unique) * val_frac)
    shuffled = unique[np.random.default_rng(seed).permutation(len(unique))]
    return set(shuffled[:n_val])


def _check_use_values(use: pd.Series) -> None:
    """Refuse any `use` value the published split does not define."""
    unexpected = sorted(set(use) - set(USE_VALUES))
    if unexpected:
        raise ValueError(
            f"unexpected {USE_COLUMN!r} values {unexpected}; "
            f"the published split defines only {list(USE_VALUES)}"
        )


def assign_splits(
    table: pd.DataFrame, seed: int = SEED, val_frac: float = VAL_FRAC
) -> np.ndarray:
    """Per-row `train` / `val` / `test`, from the `use` column plus the carve.

    `test` is the `use` column verbatim, via the loader's own
    `is_published_test`; `val` is the grouped carve-out of the `Training`
    rows; everything else is `train`.
    """
    _check_use_values(table[USE_COLUMN])
    is_test = is_published_test(table).to_numpy()

    row_protospacers = protospacers(table)
    validation = carve_validation_protospacers(
        row_protospacers[~is_test], seed=seed, val_frac=val_frac
    )
    is_val = row_protospacers.isin(validation).to_numpy() & ~is_test

    return np.where(is_test, TEST, np.where(is_val, VAL, TRAIN))


def build_split(
    table: pd.DataFrame, seed: int = SEED, val_frac: float = VAL_FRAC
) -> pd.DataFrame:
    """One row per frame row: its split plus the metadata later steps need.

    `table` is expected to be the at-least-one-head frame
    (`filter_any_head`). Row order is the spreadsheet's.
    """
    table = table.reset_index(drop=True)
    row_protospacers = protospacers(table)
    library = pam_library_protospacers(table[SEQUENCE_COLUMN])

    split = pd.DataFrame(
        {
            ROW_INDEX_COLUMN: np.arange(len(table)),
            SEQUENCE_COLUMN: table[SEQUENCE_COLUMN].to_numpy(),
            PROTOSPACER_COLUMN: row_protospacers.to_numpy(),
            PAM_LIBRARY_COLUMN: row_protospacers.isin(library).to_numpy(),
            TRNA_COLUMN: table[TRNA_COLUMN].to_numpy(),
            USE_COLUMN: table[USE_COLUMN].to_numpy(),
            SPLIT_COLUMN: assign_splits(table, seed=seed, val_frac=val_frac),
        }
    )
    for head in HEAD_NAMES:
        split[head] = table[head].to_numpy(dtype=float)
    return split[OUTPUT_COLUMNS]


def partition(split: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The split frame cut into its three parts, each in source row order."""
    return {
        name: split[split[SPLIT_COLUMN] == name].reset_index(drop=True)
        for name in SPLITS
    }


def _counts(parts: dict[str, pd.DataFrame]) -> dict:
    """Row and protospacer counts for each named frame."""
    return {
        "rows": {name: len(part) for name, part in parts.items()},
        "protospacers": {
            name: int(part[PROTOSPACER_COLUMN].nunique())
            for name, part in parts.items()
        },
    }


def summarize(split: pd.DataFrame) -> dict:
    """Counts per split, plus the published `Use`-column counts.

    `rows`/`protospacers` are the three splits we hand to training, so their
    `train` is the **post-carve** figure. `published` re-states the same rows
    cut the way the spreadsheet cuts them - train is train+val - which is what
    ADR 0004 and `CONTEXT.md` quote (32,810 / 5,186 rows, 20,759 train
    protospacers). Both are pinned so neither has to be re-derived.
    """
    is_test = is_published_test(split)
    return {
        **_counts({**partition(split), "total": split}),
        "published": _counts(
            {
                TRAIN: split[~is_test],
                TEST: split[is_test],
                "total": split,
            }
        ),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_split(
    split: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    seed: int = SEED,
    dataset_path: Path = DATASET_PATH,
) -> dict:
    """Write the three CSVs and the checksum sidecar; return the record."""
    output_dir.mkdir(parents=True, exist_ok=True)

    digests = {}
    for name, part in partition(split).items():
        path = output_dir / SPLIT_FILENAMES[name]
        # Fixed line ending so the digests match across Windows and Colab.
        part.to_csv(path, index=False, lineterminator="\n")
        digests[SPLIT_FILENAMES[name]] = file_sha256(path)

    counts = summarize(split)
    record = {
        "seed": seed,
        "dataset_sha256": file_sha256(dataset_path),
        **counts,
        "files": digests,
    }
    # Fixed line ending here too, so the sidecar itself is byte-reproducible
    # across Windows and Colab - it is the one file of the set that git tracks.
    with open(output_dir / CHECKSUM_FILENAME, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
    return record


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    # ADR 0004: the published split is a fixed artifact, not something to
    # regenerate under another seed. A sweep is fine, but not on top of the
    # one sidecar git tracks.
    if args.seed != SEED and args.output_dir == OUTPUT_DIR:
        parser.error(
            f"--seed {args.seed} would overwrite the published split in "
            f"{OUTPUT_DIR}, which ADR 0004 pins to seed {SEED}; pass "
            "--output-dir to write a different-seed split somewhere else"
        )

    split = build_split(filter_any_head(load_spcas9_table()), seed=args.seed)
    record = write_split(split, output_dir=args.output_dir, seed=args.seed)

    for name in (*SPLITS, "total"):
        print(
            f"{name:>5}: {record['rows'][name]:>6} rows, "
            f"{record['protospacers'][name]:>6} protospacers"
        )
    published = record["published"]
    print(
        f"published Use column: {published['rows'][TRAIN]} train "
        f"({published['protospacers'][TRAIN]} protospacers) / "
        f"{published['rows'][TEST]} test"
    )
    print(f"Dataset SHA256: {record['dataset_sha256']}")
    print(f"Wrote {args.output_dir / CHECKSUM_FILENAME} and the three split CSVs")
    return record


if __name__ == "__main__":
    main()
