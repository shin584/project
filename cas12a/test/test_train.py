"""Parity of the ported training path with the notebook it came from (#93).

Training itself cannot be compared: it is stochastic, and the notebook ran on a
Colab GPU while this path runs on a local CPU, where the kernels do not agree
bit for bit. So the port is pinned on the three parts that *are* deterministic:

- **preprocessing** - already verified bit-exact against the pre-#91 notebook on
  all 15,000 usable rows, so it is not re-measured here; `test_data.py` and
  `test_features.py` hold the loader and feature contracts.
- **the split** - `random_state=42` makes the fold indices exact, below.
- **the structure** - layer list and parameter count, in `test_model.py`.

These tests import TensorFlow, which `cas12a.data` and `cas12a.features`
deliberately do not; they skip rather than fail where it is absent, so the
TF-free subtree stays testable on its own.
"""

import hashlib

import numpy as np
import pytest

# TensorFlow is optional for this subtree, so the imports below the skip stay
# below it - E402 is not enabled here, and moving them up would defeat it.
pytest.importorskip("tensorflow")

from cas12a.train import (
    BATCH_SIZE,
    EPOCHS,
    N_SPLITS,
    RANDOM_STATE,
    SHUFFLE,
    make_kfold,
)

#: Usable rows in `kim_2018_cas12a_dataset.xlsx`: 15,002 rows less the two whose
#: indel frequency is missing. The split is a function of this count alone, so
#: the golden values below need no data file.
USABLE_ROWS = 15000

#: SHA-256 of each fold's validation indices as int64, in fold order, for
#: `KFold(5, shuffle=True, random_state=42).split(range(15000))`. A digest
#: rather than 3,000 literals; the first few indices are asserted separately so
#: a failure says something readable before it says "hash mismatch".
FOLD_VAL_DIGESTS = (
    "72665cea4962293e8bcc0c133bd6f78d8c98d376c7694d47c7dc6dfc52be3fc1",
    "b80cf3d00924728bf67c992708b0bae263485360053176840e3402ba96fc3174",
    "809581dd5c856440de0e7c384860993b76734319c473cbdb837be5f1313442dc",
    "22bd88f665be1219482a0bf1bdb908972202b431b0c93a6f7956955603b13a8b",
    "8a6e1e6083c1afdb9fbc0cec5e537a110a357f40d778fde66adb1d05efa46e7f",
)

#: The head of each fold's validation indices, same split.
FOLD_VAL_HEADS = (
    (3, 8, 14, 15, 19),
    (0, 10, 17, 23, 29),
    (12, 20, 26, 28, 30),
    (2, 6, 7, 18, 22),
    (1, 4, 5, 9, 11),
)


def _folds(n_rows=USABLE_ROWS):
    return list(make_kfold().split(np.arange(n_rows)))


def test_splitter_config_is_the_notebook_s():
    """The three values that make the split reproducible, pinned.

    Changing any of them changes which rows a fold validates on, which makes
    every number reported before and after incomparable. #92 may raise the
    epoch count; it must not touch these.
    """
    kf = make_kfold()
    assert (kf.n_splits, kf.shuffle, kf.random_state) == (5, True, 42)
    assert (N_SPLITS, SHUFFLE, RANDOM_STATE) == (5, True, 42)


def test_training_configuration_is_unchanged():
    """`epochs=5` and `batch_size=32`, which #93 ports rather than fixes.

    Five epochs is the review's P2 - the model is under-trained - and raising
    it is #92's call, made with metrics in hand. This test is here so that
    raising it is a deliberate edit to a test rather than a silent one.
    """
    assert (EPOCHS, BATCH_SIZE) == (5, 32)


def test_fold_indices_match_the_notebook_split():
    folds = _folds()
    assert len(folds) == N_SPLITS

    for fold, (_, val_idx) in enumerate(folds):
        digest = hashlib.sha256(np.asarray(val_idx, dtype=np.int64).tobytes())
        assert tuple(val_idx[:5]) == FOLD_VAL_HEADS[fold], f"fold {fold} head"
        assert digest.hexdigest() == FOLD_VAL_DIGESTS[fold], f"fold {fold} digest"


def test_train_5fold_runs_without_colab_and_reports_nothing():
    """The whole loop, on 25 rows, with no Drive mount anywhere in reach.

    Two things at once. It is the end-to-end check that #93 asked for - the
    training path completes inside `cas12a/` on a CPU - and it pins the
    behaviour the port had to keep: `fold_results` comes back **empty**, so a
    caller gets five trained models and not one number about any of them. #92
    fills it; until then this test says the gap is deliberate.
    """
    from cas12a.train import train_5fold

    rows = 25
    rng = np.random.default_rng(0)
    X_seq = rng.random((rows, 34, 4), dtype=np.float32)
    X_feat = rng.random((rows, 354), dtype=np.float32)
    y = rng.random(rows, dtype=np.float32)

    fold_models, fold_results = train_5fold(X_seq, X_feat, y)

    assert len(fold_models) == N_SPLITS
    assert fold_results == []


def test_folds_partition_every_row_exactly_once():
    """No row is validated twice and none is left out.

    Cheap, but it is the property that makes 5-fold cross-validation mean what
    it says, and it would catch a splitter swapped for one that samples.
    """
    folds = _folds()
    validated = np.concatenate([val_idx for _, val_idx in folds])

    assert sorted(validated) == list(range(USABLE_ROWS))
    for train_idx, val_idx in folds:
        assert len(train_idx) == USABLE_ROWS - len(val_idx)
        assert not set(train_idx) & set(val_idx)
