"""Score measured 9-Hydra rows, including NGG populations (issues #48, #82).

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
  rows back to the split without re-deriving any of them. That key is the
  split's and only the split's: it counts the at-least-one-head frame, while
  the v1 membership rebuild numbers the all-9-heads frame, so the two
  `row_index` columns do **not** address the same rows. v1's categories are a
  property of the protospacer, so a comparison reaches them through
  `protospacer` / `sequence` - never by joining the two row indices. The
  sidecar pins the CSV's digest, the checkpoint's, and the consumed
  **9-Hydra split**'s.

Metrics are on the scale v2 was trained on: the published percentage divided
by 100.

The split is read through `spcas9_train.load_published_split`, so scoring
verifies the CSVs against `split_checksum.json` the same way training did -
a checkpoint and its score cannot come from two different splits. Nothing
here retrains or re-splits.

NGG and non-NGG are sequence-derived populations, separate from the existing
PAM-library and tRNA strata. Shared population arithmetic lives in
`spcas9_metrics` for reuse by validation monitors and comparison reports.

Use `--predictions` to verify and re-evaluate saved predictions without loading
a model; otherwise predict from the checkpoint on `--split val|test`. Test is
restricted to the independently pinned historical v2 checkpoint, predictions
and published inputs. Candidate Test remains behind the study finalist guard.

Inputs are read-only. The output directory must be new, including when the
default historical destination already exists.

Usage: python -m SpCas9.spcas9_eval --output-dir NEW_DIR [--predictions CSV]
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from io import BytesIO
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
from SpCas9.spcas9_execution import reserve_destination, source_record
from SpCas9.spcas9_metrics import (
    equal_head_ngg_summary,
    head_metrics,
    population_metrics,
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
    VAL,
    file_sha256,
    write_json,
)
from SpCas9.spcas9_train import CHECKPOINT_FILENAME, split_record
from SpCas9.spcas9_train import OUTPUT_DIR as TRAIN_DIR

CHECKPOINT_PATH = TRAIN_DIR / CHECKPOINT_FILENAME
OUTPUT_DIR = BASE_DIR / "SpCas9" / "v2_eval"

PREDICTIONS_FILENAME = "predictions.csv"
PREDICTIONS_CHECKSUM_FILENAME = "predictions_checksum.json"
RESULTS_FILENAME = "results.json"

# Existing, already evaluated v2 only. New candidates must use the study's
# guarded finalist/Test stage; this historical evaluator cannot authorize it.
STORED_V2_SHA256 = "ece5009215482d3fd2506688dc8e2423437abdcd28dfe10e4ac21c5bed3d19d6"
STORED_PREDICTIONS_SHA256 = (
    "2a6d9a70993787c839afe48d0489b8d1542003f0c817a56ac94f35f97bbe296b"
)

# The split columns that ride along with the predictions: `row_index` joins a
# row back to the split, and `protospacer` is what v1's categories are defined
# over, so neither has to be re-derived downstream.
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
# The tRNA feature's two values. The report keys its strata by `str(value)`,
# since JSON has no integer keys: a reader of `results.json` and a caller
# holding the dict in memory then index a stratum the same way. The row
# comparison stays numeric, so the cut does not depend on how pandas typed
# the column.
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
    `measured_rows` reads to keep a head's metrics to its own rows - while every
    row carries a prediction, since the model emits all nine heads for every
    sequence.
    """
    metadata = METADATA_COLUMNS + [
        column
        for column in test_frame.columns
        if column not in METADATA_COLUMNS and column not in HEAD_NAMES
    ]
    frame = test_frame[metadata].copy().reset_index(drop=True)
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


def measured_rows(frame: pd.DataFrame, head: str) -> pd.DataFrame:
    """The rows of `frame` on which `head`'s variant was actually measured."""
    return frame[frame[true_column(head)].notna()]


def strata_metrics(frame: pd.DataFrame, head: str) -> dict:
    """`head`'s metrics cut by PAM-library membership and by tRNA feature.

    Both cuts are taken from the split's own metadata columns, and both over
    `measured_rows` only, so each stratum's `n` adds up to the head's own.
    """
    rows = measured_rows(frame, head)
    in_library = rows[PAM_LIBRARY_COLUMN].to_numpy(dtype=bool)
    return {
        PAM_LIBRARY_COLUMN: {
            PAM_LIBRARY: head_metrics(rows[in_library], head),
            NON_LIBRARY: head_metrics(rows[~in_library], head),
        },
        TRNA_COLUMN: {
            str(value): head_metrics(rows[rows[TRNA_COLUMN] == value], head)
            for value in TRNA_VALUES
        },
    }


def empty_strata(strata: dict) -> list[str]:
    """`<stratum>=<value>` for every group of `strata` holding no rows.

    An empty stratum is not a result: it means this head was never measured
    on those rows, and saying so keeps an n=0 entry from reading as a
    comparison that came out even. Takes the report `strata_metrics` already
    built, so a head's strata are computed once.
    """
    return [
        f"{name}={value}"
        for name, groups in strata.items()
        for value, metrics in groups.items()
        if metrics["n"] == 0
    ]


def head_report(frame: pd.DataFrame, head: str) -> dict:
    """`head`'s headline metrics over its measured rows, plus the strata."""
    report = head_metrics(measured_rows(frame, head), head)
    report["strata"] = strata_metrics(frame, head)
    return report


def build_report(frame: pd.DataFrame, *, metrics_schema_version: int = 2) -> dict:
    """Scoring report; schema 1 is retained to verify intact historical studies."""
    if metrics_schema_version not in (1, 2):
        raise ValueError("unsupported metrics schema version")
    heads = {head: head_report(frame, head) for head in HEAD_NAMES}
    report = {
        "model": "9-Hydra v2",
        "testset": "9-Hydra Testset",
        "scale": SCALE_NOTE,
        "rows": len(frame),
        "heads": heads,
        "empty_strata": {
            head: empty_strata(report["strata"]) for head, report in heads.items()
        },
    }
    if metrics_schema_version == 1:
        for scored in heads.values():
            del scored["unavailable_reasons"]
            for groups in scored["strata"].values():
                for metrics in groups.values():
                    del metrics["unavailable_reasons"]
    else:
        populations = population_metrics(frame)
        report.update(
            metrics_schema_version=2,
            populations=populations,
            equal_head_ngg=equal_head_ngg_summary(populations),
        )
    return report


def write_predictions(
    frame: pd.DataFrame,
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
    split_dir: Path = SPLIT_DIR,
    *,
    input_predictions: dict | None = None,
) -> dict:
    """Write the predictions CSV and its checksum sidecar; return the record.

    The sidecar is what lets #49 cite these predictions without reloading
    TensorFlow: it pins the CSV's own digest, the scored checkpoint's, and the
    consumed split's (through `split_record`, which also carries the
    spreadsheet's `dataset_sha256`).
    """
    for filename in (
        PREDICTIONS_FILENAME,
        PREDICTIONS_CHECKSUM_FILENAME,
        RESULTS_FILENAME,
    ):
        path = output_dir / filename
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    checkpoint_digest = file_sha256(checkpoint_path)
    split = split_record(split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / PREDICTIONS_FILENAME
    # Fixed line ending so `file_sha256` matches across Windows and Colab.
    frame.to_csv(predictions_path, index=False, lineterminator="\n")

    record = {
        "model": "9-Hydra v2",
        "rows": len(frame),
        "scale": SCALE_NOTE,
        "coverage": {head: len(measured_rows(frame, head)) for head in HEAD_NAMES},
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_digest,
        "split": split,
        "file": PREDICTIONS_FILENAME,
        "file_sha256": file_sha256(predictions_path),
    }
    if input_predictions is not None:
        record.update(input_predictions)
    write_json(record, output_dir / PREDICTIONS_CHECKSUM_FILENAME)
    return record


def validate_prediction_rows(frame: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Require saved predictions to retain the split's rows, metadata and labels."""
    try:
        expected = build_predictions(
            rows, {head: frame[pred_column(head)].to_numpy() for head in HEAD_NAMES}
        )
        pd.testing.assert_frame_equal(
            frame[expected.columns],
            expected,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-14,
        )
    except (AssertionError, KeyError) as error:
        raise ValueError(
            "predictions do not match split rows, metadata or labels"
        ) from error


def _read_predictions(
    path: Path, sidecar: Path, rows: pd.DataFrame, checkpoint_digest: str, split: dict
) -> tuple[pd.DataFrame, dict]:
    """Verify saved bytes and their row/label alignment without running a model."""
    raw, sidecar_raw = path.read_bytes(), sidecar.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    record = json.loads(sidecar_raw)
    if (
        record["file_sha256"] != digest
        or record["checkpoint_sha256"] != checkpoint_digest
        or record["split"] != split
    ):
        raise ValueError("prediction/checkpoint/split SHA-256 evidence differs")
    frame = pd.read_csv(BytesIO(raw))
    if record["rows"] != len(frame):
        raise ValueError("prediction sidecar row count differs")
    validate_prediction_rows(frame, rows)
    return frame, {
        "input_predictions": str(path.resolve()),
        "input_predictions_sha256": digest,
        "input_sidecar": str(sidecar.resolve()),
        "input_sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--split", choices=(VAL, TEST), default=TEST)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Re-evaluate saved predictions without inference",
    )
    parser.add_argument("--prediction-sidecar", type=Path)
    args = parser.parse_args(argv)

    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    if args.prediction_sidecar and not args.predictions:
        parser.error("--prediction-sidecar requires --predictions")
    checkpoint_digest = file_sha256(args.checkpoint)
    if args.split == TEST:
        if checkpoint_digest != STORED_V2_SHA256:
            raise PermissionError(
                "candidate Test evaluation requires the guarded study finalist stage"
            )
        if (
            args.predictions
            and file_sha256(args.predictions) != STORED_PREDICTIONS_SHA256
        ):
            raise PermissionError(
                "Test replay is restricted to the stored v2 predictions"
            )
    from SpCas9.spcas9_study_inputs import StudyInputs

    splits = StudyInputs(split_dir=args.split_dir).validate()
    rows = splits[args.split]
    split = split_record(args.split_dir)
    input_predictions = None
    sources: tuple[Path, ...] = (args.checkpoint, args.split_dir)
    if args.predictions:
        sidecar = args.prediction_sidecar or args.predictions.with_name(
            PREDICTIONS_CHECKSUM_FILENAME
        )
        frame, input_predictions = _read_predictions(
            args.predictions, sidecar, rows, checkpoint_digest, split
        )
        sources += (args.predictions, sidecar)
    else:
        preds = score_checkpoint(rows[SEQUENCE_COLUMN].tolist(), args.checkpoint)
        frame = build_predictions(rows, preds)
    if (
        file_sha256(args.checkpoint) != checkpoint_digest
        or split_record(args.split_dir) != split
    ):
        raise ValueError("checkpoint or split changed during evaluation")
    report = build_report(frame)
    report["split"] = args.split
    if args.split != TEST:
        del report["testset"]
    report["execution"] = {
        "mode": "saved_predictions" if args.predictions else "checkpoint_inference",
        "interpreter": sys.executable,
        "source": source_record(),
    }
    reserve_destination(args.output_dir, sources)
    checksum = write_predictions(
        frame,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        input_predictions=input_predictions,
    )

    report["predictions"] = checksum
    report["libraries"] = library_versions()
    results_path = write_json(report, args.output_dir / RESULTS_FILENAME)

    for head, scored in report["heads"].items():
        print(
            f"{head:>13}: n={scored['n']:>5} "
            f"Spearman {scored['spearman']} Pearson {scored['pearson']} "
            f"MAE {scored['mae']} MSE {scored['mse']}"
        )
    print(f"Equal-head NGG: {report['equal_head_ngg']}")
    print(f"Wrote {args.output_dir / PREDICTIONS_FILENAME}, {results_path}")
    return report


if __name__ == "__main__":
    main()
