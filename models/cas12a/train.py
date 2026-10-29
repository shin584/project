"""5-fold training entry point for the Cas12a model.

Run it as `python -m cas12a.train`. #93 moved this path out of
`cas12a/model/Cas12a_model_final.ipynb` - no Drive mount, no `/content/...`
path - and deliberately left the training configuration broken so that a later
performance change could be attributed to the retune and not to the port. #92
is that retune, and this module is now where the review's P1, P2 and P4 are
answered (`cas12a/docs/cas12a-model-review.md` §3):

- **P1, no metrics at all.** Every fold is scored - Spearman, Pearson, R², MAE,
  MSE, plus calibration - by `cas12a/metrics.py`, and `fold_results` comes back
  full. The split is untouched: `KFold(5, shuffle=True, random_state=42)`, so
  these numbers are comparable to §2's ridge baseline row for row.
- **P2, under-training with no best checkpoint.** `EarlyStopping` and
  `ReduceLROnPlateau` are attached to `model.fit` instead of merely imported,
  `restore_best_weights=True` means the weights that get scored are the best
  epoch's and not whichever one the loop happened to end on, and the epoch cap
  is `EPOCHS` rather than 5. The cost measurement in
  `cas12a/docs/cas12a-training-cost.md` is why that is affordable locally.
- **P4, nothing saved.** The best fold's `.keras` and the per-fold metrics JSON
  are written to the same directory, so the file says which numbers it scored.

Seeds are fixed (#92 task 4) but `tf.config.experimental.enable_op_determinism`
is **not** called. The gate is a relative comparison in units of the fold
standard deviation, which already absorbs run-to-run jitter smaller than
itself; strict op determinism would only make training slower.

The console log is not an artifact. Per-epoch Keras output is noise a later
reader should not have to page through - `cas12a/metrics.py` writes the JSON
that is the single source of these numbers.
"""

import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from cas12a.data import load_cas12a_dataset
from cas12a.metrics import (
    GATE_KEY,
    aggregate,
    fold_metrics,
    format_summary,
    gate_verdict,
)
from cas12a.model import create_cas12a_model

#: The notebook's 5-fold cross-validation, pinned. Changing any of the three
#: changes which rows a fold validates on, which would make these metrics
#: incomparable to the ridge baseline they are gated against.
N_SPLITS = 5
SHUFFLE = True
RANDOM_STATE = 42

#: Epoch **ceiling**, not a target: `EarlyStopping` decides where a fold
#: actually stops. Five was the notebook's and is the review's P2; at ~10s an
#: epoch on CPU a ceiling this high costs nothing when early stopping fires,
#: and the convergence trace in `cas12a/docs/cas12a-training-cost.md` shows
#: four of five folds were still improving when the notebook cut them off.
EPOCHS = 100
BATCH_SIZE = 32

#: `restore_best_weights` is the half of P2 that mattered most: without it the
#: scored weights are the last epoch's, so a reported number depended on where
#: the loop stopped. `patience` is generous because val_loss on this dataset
#: moves in the fourth decimal.
EARLY_STOPPING_MONITOR = "val_loss"
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-5

#: Seeds Python, NumPy and TensorFlow. Per-fold the seed is `SEED + fold`, so
#: re-running one fold reproduces that fold rather than the first one.
SEED = 42

#: Default dataset, resolved against the package so the entry point works from
#: any working directory.
DEFAULT_DATASET = (
    Path(__file__).resolve().parent / "dataset" / "kim_2018_cas12a_dataset.xlsx"
)

#: Where the `.keras` and its metrics JSON land, together. `.gitignore` tracks
#: the JSON and ignores the checkpoint, the same split 9-Hydra's `training_log.
#: json` uses: the numbers are small and are the reproducibility record, the
#: weights are large and only matter once something decides to adopt them.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
METRICS_FILENAME = "cas12a_5fold_metrics.json"


def make_kfold():
    """The pinned splitter, as its own call so a test can assert the config."""
    return KFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)


def set_seeds(seed):
    """Seed Python, NumPy and TensorFlow together.

    `tf.keras.utils.set_random_seed` does all three in one call, which is the
    point of using it over three separate ones - weight initialisation, dropout
    masks and the batch shuffle all draw from seeded generators. It does not
    make training bit-reproducible on its own, and nothing here claims it does:
    op determinism is deliberately left off (see the module docstring), so two
    runs with the same seed land close rather than identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def library_versions():
    """The versions these numbers were produced with, for the JSON record.

    `pandas`, `scipy` and `sklearn` are imported here rather than at module
    scope because this function is the only thing in the module that wants
    them - every one is already loaded by the time it is called, through
    `cas12a.data` and `cas12a.metrics`.
    """
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "tensorflow": tf.__version__,
    }


def make_callbacks():
    """The two callbacks the notebook imported and never attached (P2)."""
    return [
        EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=EARLY_STOPPING_MONITOR,
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1,
        ),
    ]


def training_configuration():
    """The settings that produced a run, as the JSON records them."""
    return {
        "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "early_stopping": {
            "monitor": EARLY_STOPPING_MONITOR,
            "patience": EARLY_STOPPING_PATIENCE,
            "restore_best_weights": True,
        },
        "reduce_lr_on_plateau": {
            "monitor": EARLY_STOPPING_MONITOR,
            "factor": REDUCE_LR_FACTOR,
            "patience": REDUCE_LR_PATIENCE,
            "min_lr": REDUCE_LR_MIN,
        },
        "op_determinism_enabled": False,
    }


def train_5fold(X_seq, X_feat, y, epochs=EPOCHS, seed=SEED, verbose=2):
    """Fit one freshly initialised model per fold and score every one of them.

    Returns `(fold_models, fold_results)`. `fold_results` is a list of dicts -
    one per fold, in fold order - carrying the metrics from
    `cas12a.metrics.fold_metrics` plus how the fold trained: `epochs_ran`,
    `best_epoch` and `best_val_loss`. It used to come back empty (the review's
    P1); a caller now gets a number for every fold it got a model for.

    `best_epoch` is read off the history rather than from the callback, because
    it has to describe the weights that are actually in the returned model -
    `restore_best_weights=True` puts back the minimum-`val_loss` epoch, which
    is exactly `argmin` of the recorded `val_loss`.

    The parameter names are the notebook's, capitals included, so that a call
    written against the notebook moves here unchanged.
    """
    kf = make_kfold()
    fold_results, fold_models = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_seq)):
        print(f"\n--- Fold {fold + 1} Training ---", flush=True)
        set_seeds(seed + fold)
        model = create_cas12a_model()

        started = time.perf_counter()
        history = model.fit(
            [X_seq[train_idx], X_feat[train_idx]],
            y[train_idx],
            validation_data=([X_seq[val_idx], X_feat[val_idx]], y[val_idx]),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=make_callbacks(),
            verbose=verbose,
        )
        elapsed = time.perf_counter() - started

        val_loss = history.history["val_loss"]
        best_epoch = int(np.argmin(val_loss)) + 1
        y_pred = model.predict(
            [X_seq[val_idx], X_feat[val_idx]], batch_size=BATCH_SIZE, verbose=0
        )

        result = {
            "fold": fold + 1,
            "val_rows": len(val_idx),
            "epochs_ran": len(val_loss),
            "best_epoch": best_epoch,
            "best_val_loss": float(val_loss[best_epoch - 1]),
            "final_val_loss": float(val_loss[-1]),
            "wall_clock_s": round(elapsed, 1),
            **fold_metrics(y[val_idx], y_pred),
        }
        fold_results.append(result)
        fold_models.append(model)

        print(
            f"--- Fold {fold + 1}: best epoch {best_epoch}/{len(val_loss)}, "
            f"val_loss {result['best_val_loss']:.4f}, "
            f"Spearman {result['spearman']:.4f}, MAE {result['mae']:.4f} "
            f"({elapsed:.1f}s) ---",
            flush=True,
        )

    return fold_models, fold_results


def best_fold_index(fold_results):
    """Index of the fold with the highest validation Spearman.

    Spearman because that is the gate metric, so the saved checkpoint is the
    one the adoption decision was made on. NaN-safe: a collapsed fold is
    skipped rather than winning by comparing False against everything.
    """
    scores = np.asarray([fold[GATE_KEY] for fold in fold_results], dtype=np.float64)
    if np.all(np.isnan(scores)):
        raise ValueError(f"every fold's {GATE_KEY} is NaN; nothing to save")
    return int(np.nanargmax(scores))


def save_artifacts(fold_models, fold_results, output_dir, dataset, rows):
    """Write the best fold's `.keras` and the per-fold metrics JSON together.

    Together is the requirement (#92 task 5). What matters is not bit-identical
    retraining - seeds are fixed but op determinism is off - but that the
    checkpoint on disk can be traced to the numbers it scored, which the JSON
    holds beside it under `saved_model`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best = best_fold_index(fold_results)
    best_result = fold_results[best]
    model_filename = f"cas12a_best_fold{best_result['fold']}.keras"
    fold_models[best].save(output_dir / model_filename)

    summary = aggregate(fold_results)
    verdict = gate_verdict(summary)
    record = {
        "issue": 92,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"path": str(dataset), "rows": int(rows)},
        "split": {
            "n_splits": N_SPLITS,
            "shuffle": SHUFFLE,
            "random_state": RANDOM_STATE,
        },
        "training": training_configuration(),
        "library_versions": library_versions(),
        "folds": fold_results,
        "aggregate": summary,
        "gate": verdict,
        "saved_model": {
            "filename": model_filename,
            "fold": best_result["fold"],
            "selected_by": f"highest validation {GATE_KEY}",
            GATE_KEY: best_result[GATE_KEY],
            "mae": best_result["mae"],
            "best_val_loss": best_result["best_val_loss"],
        },
    }

    metrics_path = output_dir / METRICS_FILENAME
    metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return metrics_path, output_dir / model_filename, summary, verdict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Kim 2018 table to train on (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where the .keras and metrics JSON go (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            f"Epoch ceiling; early stopping usually ends a fold sooner "
            f"(default: {EPOCHS}). Lower it for a smoke run - the result is "
            f"not a reportable measurement."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    set_seeds(SEED)
    X_seq, X_feat, y = load_cas12a_dataset(args.dataset)
    print(f"Loaded {len(y)} usable rows from {args.dataset}", flush=True)

    started = time.perf_counter()
    fold_models, fold_results = train_5fold(X_seq, X_feat, y, epochs=args.epochs)
    elapsed = time.perf_counter() - started

    metrics_path, model_path, summary, verdict = save_artifacts(
        fold_models, fold_results, args.output_dir, args.dataset, len(y)
    )

    print("\n".join(format_summary(fold_results, summary, verdict)), flush=True)
    print(f"\nTrained {len(fold_models)} folds in {elapsed:.1f}s wall-clock.")
    print(f"Metrics: {metrics_path}")
    print(f"Checkpoint: {model_path}")
    if args.epochs != EPOCHS:
        print(
            f"NOTE: --epochs {args.epochs} overrode the {EPOCHS}-epoch ceiling; "
            "this run is not a reportable measurement."
        )
    return fold_models, fold_results


if __name__ == "__main__":
    main()
