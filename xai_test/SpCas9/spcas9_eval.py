"""Score 9-Hydra v2 on the 9-Hydra Testset, per head and by stratum (issue #48).

Where 9-Hydra v2 stands on the 5,186-row **9-Hydra Testset**, scored once and
written down so the v1 <-> v2 comparison (#49) reads the predictions rather
than re-running the model.

- **Per head: n / Spearman / Pearson / MAE / MSE over that head's available
  test rows only.** A head reads `n.a.` on rows where its variant was never
  measured, and those rows are dropped from that head's metrics exactly as
  the **head mask** drops them from the loss - no metric is an average over
  fabricated zeros. Coverage is the pinned 5,021 (`SpCas9`), ~5,170 (most
  heads) and 3,982 / 3,979 (`SpCas9-NG` / `VRQR variant`).
- **Two strata, both read from the split metadata rather than recomputed:**
  `in_pam_library` (PAM-library vs non-library rows) and the **tRNA feature**
  (0 vs 1). The split is the single source of truth for both - re-deriving
  the PAM library here could silently disagree with the rows v2 trained on.
  The tRNA stratum exists to show whether feeding that feature to v3 would be
  worth building; it stays out of the model, as in v1.
- **A stratum with no rows is named, not silently reported as a tie.**
  `SpCas9-NG` and `VRQR variant` have no tRNA=1 measurement at all, so for
  those two heads the tRNA stratum carries no comparison; `empty_strata` says
  so per head rather than leaving two n=0 entries to be read as a null
  result.
- **`predictions.csv` plus `predictions_checksum.json`**, keyed by the
  split's own `row_index` and carrying its metadata columns, so #49 joins
  rows back to the split (and to v1's categories) without re-deriving
  anything. The sidecar pins the CSV's digest, the checkpoint's, and the
  consumed **9-Hydra split**'s.

Metrics are on the scale v2 was trained on: the published percentage divided
by 100.

The split is read through `spcas9_train.load_published_split`, so scoring
verifies the CSVs against `split_checksum.json` the same way training did -
a checkpoint and its score cannot come from two different splits. Nothing
here retrains or re-splits.

Full scoring is not unit-tested; running the script is the check, as with #38.

Read-only with respect to the checkpoint and every dataset file; writes only
the three files in `SpCas9/v2_eval/`.

Usage: python -m SpCas9.spcas9_eval [--checkpoint ...] [--output-dir ...]
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from SpCas9.spcas9_data import (
    BASE_DIR,
    HEAD_NAMES,
    SEQUENCE_COLUMN,
    TRNA_COLUMN,
    USE_COLUMN,
    head_labels,
    library_versions,
    one_hot,
)
from SpCas9.spcas9_metrics import (
    full_metrics,
    head_arrays,
    pred_column,
    predictions_by_head,
    true_column,
)
from SpCas9.spcas9_split import OUTPUT_DIR as SPLIT_DIR
from SpCas9.spcas9_split import (
    PAM_LIBRARY_COLUMN,
    PROTOSPACER_COLUMN,
    ROW_INDEX_COLUMN,
    SPLIT_COLUMN,
    TEST,
    file_sha256,
)
from SpCas9.spcas9_train import CHECKPOINT_FILENAME, load_published_split, split_record
from SpCas9.spcas9_train import OUTPUT_DIR as TRAIN_DIR

CHECKPOINT_PATH = TRAIN_DIR / CHECKPOINT_FILENAME
OUTPUT_DIR = BASE_DIR / "SpCas9" / "v2_eval"

PREDICTIONS_FILENAME = "predictions.csv"
PREDICTIONS_CHECKSUM_FILENAME = "predictions_checksum.json"
RESULTS_FILENAME = "results.json"

# The split columns that ride along with the predictions, so a row can be
# joined back to the split - and through it to v1's categories - without
# re-deriving any of them.
METADATA_COLUMNS = [
    ROW_INDEX_COLUMN,
    SEQUENCE_COLUMN,
    PROTOSPACER_COLUMN,
    PAM_LIBRARY_COLUMN,
    TRNA_COLUMN,
    USE_COLUMN,
    SPLIT_COLUMN,
]

# The two sides of the PAM-library stratum, and the tRNA feature's two values.
# Each stratum keys its groups under the metadata column it is cut from, so a
# reported figure names the column a reader can check it against.
PAM_LIBRARY = "library"
NON_LIBRARY = "non_library"
TRNA_VALUES = (0, 1)

SCALE_NOTE = "published percentage / 100 (the scale 9-Hydra v2 was trained on)"


def score_checkpoint(
    sequences: Sequence[str], checkpoint_path: Path = CHECKPOINT_PATH
) -> dict[str, np.ndarray]:
    """v2's predictions for `sequences`, keyed by head name. Not unit-tested.

    v2's functional model returns its outputs as a dict keyed by output layer
    name, which `predictions_by_head` re-keys by head name.
    """
    import keras

    model = keras.models.load_model(checkpoint_path, compile=False)
    outputs = model.predict(one_hot(sequences), batch_size=1024, verbose=0)
    return predictions_by_head(outputs)


def build_predictions(
    test_frame: pd.DataFrame, preds: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """The split's metadata columns plus each head's label and prediction.

    An unmeasured label stays missing rather than being filled - that is what
    `available` reads to keep a head's metrics on its own rows - while every
    row carries a prediction, since the model emits all nine heads for every
    sequence.
    """
    frame = test_frame[METADATA_COLUMNS].copy().reset_index(drop=True)
    labels = head_labels(test_frame)
    for head in HEAD_NAMES:
        values = np.asarray(preds[head], dtype=np.float64).reshape(-1)
        if len(values) != len(frame):
            raise ValueError(
                f"head {head!r} has {len(values)} predictions for "
                f"{len(frame)} test rows"
            )
        frame[true_column(head)] = labels[head]
        frame[pred_column(head)] = values
    return frame


def available(frame: pd.DataFrame, head: str) -> pd.DataFrame:
    """The rows of `frame` on which `head`'s variant was actually measured."""
    return frame[frame[true_column(head)].notna()]


def _metrics(rows: pd.DataFrame, head: str) -> dict:
    return full_metrics(*head_arrays(rows, head))


def strata(frame: pd.DataFrame, head: str) -> dict:
    """`head`'s metrics cut by PAM-library membership and by tRNA feature.

    Both cuts are taken from the split's own metadata columns, and both are
    taken over `available` rows only, so each stratum's `n` adds up to the
    head's own.
    """
    rows = available(frame, head)
    in_library = rows[PAM_LIBRARY_COLUMN].to_numpy(dtype=bool)
    return {
        PAM_LIBRARY_COLUMN: {
            PAM_LIBRARY: _metrics(rows[in_library], head),
            NON_LIBRARY: _metrics(rows[~in_library], head),
        },
        TRNA_COLUMN: {
            value: _metrics(rows[rows[TRNA_COLUMN] == value], head)
            for value in TRNA_VALUES
        },
    }


def empty_strata(frame: pd.DataFrame, head: str) -> list[str]:
    """`<stratum>=<value>` for every one of `head`'s strata holding no rows.

    An empty stratum is not a result: it means this head was never measured
    on those rows, and saying so keeps an n=0 entry from reading as a
    comparison that came out even.
    """
    return [
        f"{name}={value}"
        for name, groups in strata(frame, head).items()
        for value, metrics in groups.items()
        if metrics["n"] == 0
    ]


def head_report(frame: pd.DataFrame, head: str) -> dict:
    """`head`'s headline metrics over its available rows, plus the strata."""
    report = _metrics(available(frame, head), head)
    report["strata"] = strata(frame, head)
    return report


def build_report(frame: pd.DataFrame) -> dict:
    """The JSON-serialisable scoring report, per head and per stratum."""
    return {
        "model": "9-Hydra v2",
        "testset": "9-Hydra Testset",
        "scale": SCALE_NOTE,
        "rows": len(frame),
        "heads": {head: head_report(frame, head) for head in HEAD_NAMES},
        "empty_strata": {head: empty_strata(frame, head) for head in HEAD_NAMES},
    }


def write_json(record: dict, path: Path) -> Path:
    """Write `record` with a fixed `\\n` line ending and no NaN.

    Same rule as the split's checksum sidecar and the training log: a fixed
    line ending because git tracks these files, and `allow_nan=False` because
    `NaN` is not JSON - an undefined metric is `None` by the time it gets
    here, or this fails loudly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write("\n")
    return path


def write_predictions(
    frame: pd.DataFrame,
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
    split_dir: Path = SPLIT_DIR,
) -> dict:
    """Write the predictions CSV and its checksum sidecar; return the record.

    The sidecar is what lets #49 cite these predictions without reloading
    TensorFlow: it pins the CSV's own digest, the scored checkpoint's, and the
    consumed split's (through `split_record`, which also carries the
    spreadsheet's `dataset_sha256`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / PREDICTIONS_FILENAME
    # Fixed line ending so `file_sha256` matches across Windows and Colab.
    frame.to_csv(predictions_path, index=False, lineterminator="\n")

    record = {
        "model": "9-Hydra v2",
        "rows": len(frame),
        "scale": SCALE_NOTE,
        "coverage": {head: len(available(frame, head)) for head in HEAD_NAMES},
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "split": split_record(split_dir),
        "file": PREDICTIONS_FILENAME,
        "file_sha256": file_sha256(predictions_path),
    }
    write_json(record, output_dir / PREDICTIONS_CHECKSUM_FILENAME)
    return record


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    test_rows = load_published_split(args.split_dir)[TEST]
    preds = score_checkpoint(test_rows[SEQUENCE_COLUMN].tolist(), args.checkpoint)
    frame = build_predictions(test_rows, preds)
    checksum = write_predictions(
        frame,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
    )

    report = build_report(frame)
    report["predictions"] = checksum
    report["libraries"] = library_versions()
    results_path = write_json(report, args.output_dir / RESULTS_FILENAME)

    for head, head_metrics in report["heads"].items():
        library = head_metrics["strata"][PAM_LIBRARY_COLUMN]
        print(
            f"{head:>13}: n={head_metrics['n']:>5} "
            f"Spearman {head_metrics['spearman']:.3f} "
            f"Pearson {head_metrics['pearson']:.3f} "
            f"MAE {head_metrics['mae']:.4f} MSE {head_metrics['mse']:.4f} "
            f"| PAM library {library[PAM_LIBRARY]['n']:>4} rows, "
            f"Spearman {library[PAM_LIBRARY]['spearman']:.3f} "
            f"vs {library[NON_LIBRARY]['spearman']:.3f} elsewhere"
        )
    for head, names in report["empty_strata"].items():
        if names:
            print(f"{head:>13}: no rows in {', '.join(names)} - no comparison there")
    print(f"Wrote {args.output_dir / PREDICTIONS_FILENAME}, {results_path}")
    return report


if __name__ == "__main__":
    main()
