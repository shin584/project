"""5-fold training entry point for the Cas12a model.

Run it as `python -m cas12a.train`. This replaces the Colab path that
`cas12a/model/Cas12a_model_final.ipynb` was the only way in to: no Drive mount,
no `/content/...` path, and the dataset location is an argument that defaults to
`cas12a/dataset/`. See ADR 0002 for why the training path moved out of the
notebook, and the notebook stays until #92 reports adoptable metrics from here.

**The training configuration is deliberately unchanged** (#93 task 4). `epochs`
is still 5, the splitter is still `KFold(5, shuffle=True, random_state=42)`,
`EarlyStopping` and `ReduceLROnPlateau` are still not attached, and
`fold_results` still comes back empty. Every one of those is something the
review wants fixed - P1 and P2 in `cas12a/docs/cas12a-model-review.md` - and
fixing any of them here would mean a later performance change could not be
attributed to the port or to the retune. #92 owns them.

What is new is the wall-clock print: #93 task 6 needs the CPU cost of a fold to
decide whether #92 runs locally or goes back to Colab. It measures, it does not
record - `fold_results` stays as it was.
"""

import argparse
import time
from pathlib import Path

from sklearn.model_selection import KFold

from cas12a.data import load_cas12a_dataset
from cas12a.model import create_cas12a_model

#: The notebook's 5-fold cross-validation, pinned. `random_state` makes the
#: fold indices reproducible, which is the one piece of the port that can be
#: compared to the notebook exactly - training itself cannot, because CPU and
#: GPU kernels do not agree bit for bit.
N_SPLITS = 5
SHUFFLE = True
RANDOM_STATE = 42

EPOCHS = 5
BATCH_SIZE = 32

#: Default dataset, resolved against the package so the entry point works from
#: any working directory.
DEFAULT_DATASET = (
    Path(__file__).resolve().parent / "dataset" / "kim_2018_cas12a_dataset.xlsx"
)


def make_kfold():
    """The pinned splitter, as its own call so a test can assert the config."""
    return KFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)


def train_5fold(X_seq, X_feat, y):
    """Fit one freshly initialised model per fold.

    Returns `(fold_models, fold_results)`. `fold_results` is **empty**: the
    notebook never filled it and this port does not either, so nothing here
    reports how well any fold did. That is the review's P1, and it is #92's to
    fix. Read the printed Keras log for now.

    The parameter names are the notebook's, capitals included, so that a call
    written against the notebook moves here unchanged.
    """
    kf = make_kfold()
    fold_results, fold_models = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_seq)):
        print(f"\n--- Fold {fold + 1} Training ---")
        model = create_cas12a_model()
        started = time.perf_counter()
        model.fit(
            [X_seq[train_idx], X_feat[train_idx]],
            y[train_idx],
            validation_data=([X_seq[val_idx], X_feat[val_idx]], y[val_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )
        elapsed = time.perf_counter() - started
        print(
            f"--- Fold {fold + 1} wall-clock: {elapsed:.1f}s "
            f"({elapsed / EPOCHS:.1f}s/epoch) ---"
        )
        fold_models.append(model)

    return fold_models, fold_results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Kim 2018 table to train on (default: {DEFAULT_DATASET}).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    X_seq, X_feat, y = load_cas12a_dataset(args.dataset)
    print(f"Loaded {len(y)} usable rows from {args.dataset}")

    started = time.perf_counter()
    fold_models, fold_results = train_5fold(X_seq, X_feat, y)
    elapsed = time.perf_counter() - started

    print(f"\nTrained {len(fold_models)} folds in {elapsed:.1f}s wall-clock.")
    return fold_models, fold_results


if __name__ == "__main__":
    main()
