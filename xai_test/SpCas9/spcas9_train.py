"""Build the 9-Hydra v2 masked per-head training path and smoke-train it (issue #46).

The training path for **9-Hydra v2**: v1's architecture on the 9-Hydra split,
with a masked per-head loss so each row contributes only to the variant heads
it actually has measured. `--smoke` runs the whole path over one epoch and a
small subsample, so a plumbing mistake surfaces in minutes rather than hours
into the full run (#47).

- **The split is read from disk, never re-derived.** `load_published_split`
  reads `SpCas9/published_split/` and checks each CSV's SHA-256 and row count
  against `split_checksum.json` (#45), so a checkpoint is traceable to one
  split and a stale CSV is refused rather than trained on.
- **Architecture is v1's, frozen** so the dataset fix is the only moving
  variable: `Conv1D(128, 3, relu, same)` -> `BatchNormalization` ->
  `LSTM(64)` -> `Dropout(0.3)` -> `Dense(64, relu)` shared trunk, then
  per-head `Dense(32, relu)` -> `Dense(1)`. The one change is a **linear**
  output activation in place of v1's sigmoid: the labels are the published
  percentages divided by 100 and span roughly -0.075 to 0.954, and a sigmoid
  head cannot reach the negative measurements.
- **Masking is a per-output `sample_weight` dict** keyed by output layer
  name: 0 where that head reads `n.a.`, with the label filled to 0.0, and 1
  otherwise. It is passed to `weighted_metrics` as well as to the loss, so
  `val_Output_SpCas9_NG_mae` is a mean over that head's measured rows and
  never an average over fabricated zeros. A `sample_weight` dict is chosen
  over a custom NaN-masking loss precisely so the saved checkpoint reloads
  with **no custom objects registered**.
- **Outputs are a dict, where v1's were a list.** Keras 3 resolves a
  per-output `sample_weight` dict against the model's output structure, so a
  list-output model raises `KeyError: 0` the moment the weights arrive. Layer
  names, layers and weights are unchanged - only the container the functional
  model hands back - and `predict` returns a dict keyed by output name.
- **Labels are keyed by head name, never by position**, via the shared
  loader's `head_labels` and `output_name`. The spreadsheet orders
  `eSpCas9(1.1)` before `SpCas9-HF.1` and the saved model emits them the
  other way round, so positional arrays would silently swap those two heads.

Hyperparameters are v1's throughout (`Adam(0.001)`, `batch_size=32`,
`epochs=50`, `EarlyStopping(patience=5, restore_best_weights=True)`,
`ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)`), with
`keras.utils.set_random_seed(42)` added. Early stopping watches the
protospacer-grouped validation carve-out, not a Keras `validation_split`.

Reads the split artifacts and the checkpoint; writes only into its
`--output-dir`.

Usage: python -m SpCas9.spcas9_train --smoke [--smoke-rows 256] [--seed 42]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from SpCas9.spcas9_data import (
    BASE_DIR,
    HEAD_NAMES,
    SEQUENCE_COLUMN,
    clean_head_name,
    head_labels,
    one_hot,
    output_name,
)
from SpCas9.spcas9_split import (
    CHECKSUM_FILENAME,
    SPLIT_FILENAMES,
    SPLITS,
    TEST,
    TRAIN,
    VAL,
    file_sha256,
)
from SpCas9.spcas9_split import OUTPUT_DIR as SPLIT_DIR

OUTPUT_DIR = BASE_DIR / "SpCas9" / "v2_train"
SMOKE_OUTPUT_DIR = OUTPUT_DIR / "smoke"
CHECKPOINT_FILENAME = "9hydra_v2.keras"

SEQUENCE_SHAPE = (30, 4)
INPUT_LAYER_NAME = "In_Seq"

# v1's recipe, unchanged (issue #42's Implementation Decisions).
SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
LR_FACTOR = 0.2
LR_PATIENCE = 3
MIN_LR = 1e-6

# The smoke run exists to exercise the plumbing, not to fit anything.
SMOKE_EPOCHS = 1
SMOKE_ROWS = 256


def load_published_split(split_dir: Path = SPLIT_DIR) -> dict[str, pd.DataFrame]:
    """The three split CSVs, each verified against `split_checksum.json`.

    Training reads the published artifacts rather than re-deriving the split,
    so the checksum sidecar is what ties a checkpoint to one set of rows. A
    CSV whose digest or row count has moved is refused.
    """
    with open(split_dir / CHECKSUM_FILENAME, encoding="utf-8") as f:
        record = json.load(f)

    frames = {}
    for name in SPLITS:
        filename = SPLIT_FILENAMES[name]
        path = split_dir / filename
        digest = file_sha256(path)
        if digest != record["files"][filename]:
            raise ValueError(
                f"{path} has SHA-256 {digest}, but {CHECKSUM_FILENAME} pins "
                f"{record['files'][filename]}; re-run SpCas9.spcas9_split"
            )
        frame = pd.read_csv(path)
        if len(frame) != record["rows"][name]:
            raise ValueError(
                f"{path} has {len(frame)} rows, but {CHECKSUM_FILENAME} pins "
                f"{record['rows'][name]}"
            )
        frames[name] = frame
    return frames


def masked_labels_and_weights(
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-output labels and sample weights, both keyed by output layer name.

    A head that reads `n.a.` on a row gets weight 0 there and its label
    filled to 0.0; every measured head gets weight 1 and its label on the
    scale v1 trained on (the published percentage / 100). Keras needs a real
    number in the masked slot, and the weight is what keeps it out of both
    the loss and the metrics.

    Both dicts are built from `head_labels`, which keys by head name, so an
    `eSpCas9(1.1)` / `SpCas9-HF.1` transposition cannot creep in.
    """
    labels, weights = {}, {}
    for head, values in head_labels(frame).items():
        measured = np.isfinite(values)
        labels[output_name(head)] = np.where(measured, values, 0.0)
        weights[output_name(head)] = measured.astype(np.float32)
    return labels, weights


def encode_inputs(frame: pd.DataFrame) -> np.ndarray:
    """The frame's sequences as the input layer's `(n, 30, 4)` one-hot block."""
    return one_hot(frame[SEQUENCE_COLUMN].tolist())


def build_model(head_names: tuple[str, ...] = HEAD_NAMES):
    """v1's layer stack with a linear output activation on every head.

    Returns a functional model whose outputs are a **dict** keyed by output
    layer name - the structure Keras 3 resolves the per-output
    `sample_weight` dict against.
    """
    import keras

    input_seq = keras.layers.Input(shape=SEQUENCE_SHAPE, name=INPUT_LAYER_NAME)
    x = keras.layers.Conv1D(128, 3, activation="relu", padding="same")(input_seq)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LSTM(64, return_sequences=False)(x)
    x = keras.layers.Dropout(0.3)(x)
    shared_dense = keras.layers.Dense(64, activation="relu")(x)

    outputs = {}
    for head in head_names:
        clean = clean_head_name(head)
        h = keras.layers.Dense(32, activation="relu", name=f"Dense_{clean}")(
            shared_dense
        )
        # Linear, not v1's sigmoid: the labels reach below zero.
        outputs[output_name(head)] = keras.layers.Dense(1, name=f"Output_{clean}")(h)
    return keras.Model(inputs=input_seq, outputs=outputs)


def compile_model(model, learning_rate: float = LEARNING_RATE) -> None:
    """v1's `mse` loss and `mae` metric per head, Adam at `learning_rate`.

    The metric goes in `weighted_metrics`, not `metrics`: that is what makes
    the sample weights apply to the reported per-head numbers as well as to
    the gradient.
    """
    import keras

    names = list(model.output_names)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={name: "mse" for name in names},
        weighted_metrics={name: "mae" for name in names},
    )


def training_callbacks() -> list:
    """v1's early stopping and LR schedule, both watching the carve-out's loss."""
    import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
        ),
    ]


def subsample(frame: pd.DataFrame, n_rows: int, seed: int = SEED) -> pd.DataFrame:
    """A seeded `n_rows`-row sample in source row order, for the smoke run."""
    if n_rows >= len(frame):
        return frame.reset_index(drop=True)
    return frame.sample(n=n_rows, random_state=seed).sort_index().reset_index(drop=True)


def train(
    splits: dict[str, pd.DataFrame],
    output_dir: Path = OUTPUT_DIR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    """Fit the model on `splits` and save the checkpoint. Not unit-tested.

    Validation is the published carve-out passed as `validation_data` with
    its own sample weights, so early stopping is chosen on rows whose targets
    the model has not fitted - and on the same masked metrics.
    """
    import keras

    keras.utils.set_random_seed(seed)

    train_y, train_w = masked_labels_and_weights(splits[TRAIN])
    val_y, val_w = masked_labels_and_weights(splits[VAL])

    model = build_model()
    compile_model(model)
    history = model.fit(
        x=encode_inputs(splits[TRAIN]),
        y=train_y,
        sample_weight=train_w,
        validation_data=(encode_inputs(splits[VAL]), val_y, val_w),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=training_callbacks(),
        verbose=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    model.save(checkpoint_path)
    return model, history, checkpoint_path


def verify_reloads_clean(model, checkpoint_path: Path, inputs: np.ndarray) -> None:
    """Reload the checkpoint with no custom objects and match its predictions.

    The point of the `sample_weight` dict over a custom masking loss is that
    this call needs nothing registered. Comparing the reloaded predictions
    against the in-memory model's is what makes it a real check rather than
    an import that happens to succeed.
    """
    import keras

    reloaded = keras.models.load_model(checkpoint_path)
    before = model.predict(inputs, verbose=0)
    after = reloaded.predict(inputs, verbose=0)
    for name in model.output_names:
        np.testing.assert_allclose(
            np.asarray(before[name]), np.asarray(after[name]), rtol=1e-5, atol=1e-6
        )


def head_coverage(frame: pd.DataFrame) -> dict[str, int]:
    """Measured rows per head - what the masking actually keeps."""
    return {head: int(frame[head].notna().sum()) for head in HEAD_NAMES}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one epoch over a small subsample, to exercise the whole path",
    )
    parser.add_argument("--smoke-rows", type=int, default=SMOKE_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    output_dir = args.output_dir or (SMOKE_OUTPUT_DIR if args.smoke else OUTPUT_DIR)
    splits = load_published_split(args.split_dir)
    if args.smoke:
        splits = {
            name: subsample(frame, args.smoke_rows, seed=args.seed)
            for name, frame in splits.items()
        }

    for name in SPLITS:
        print(
            f"{name:>5}: {len(splits[name]):>6} rows, coverage {head_coverage(splits[name])}"
        )

    model, history, checkpoint_path = train(
        splits,
        output_dir=output_dir,
        epochs=SMOKE_EPOCHS if args.smoke else EPOCHS,
        batch_size=BATCH_SIZE,
        seed=args.seed,
    )
    verify_reloads_clean(model, checkpoint_path, encode_inputs(splits[TEST]))

    last = {key: float(values[-1]) for key, values in history.history.items()}
    print(f"loss {last['loss']:.5f} / val_loss {last['val_loss']:.5f}")
    for head in HEAD_NAMES:
        name = output_name(head)
        print(f"{head:>13}: masked val MAE {last['val_' + name + '_mae']:.5f}")
    print(f"Saved {checkpoint_path}; it reloads with no custom objects registered")
    return {"epochs_run": len(history.history["loss"]), "last_epoch": last}


if __name__ == "__main__":
    main()
