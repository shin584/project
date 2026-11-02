"""The training path: the split it inherited, and the retune #92 gave it.

Training itself cannot be compared run to run: it is stochastic, the notebook
ran on a Colab GPU while this path runs on a local CPU where the kernels do not
agree bit for bit, and #92 deliberately left op determinism off. So nothing
here asserts a performance number - `cas12a/artifacts/`'s metrics JSON reports
those. What is pinned is what is deterministic or structural:

- **preprocessing** - already verified bit-exact against the pre-#91 notebook on
  all 15,000 usable rows, so it is not re-measured here; `test_data.py` and
  `test_features.py` hold the loader and feature contracts.
- **the split** - `random_state=42` makes the fold indices exact, below, and
  #92 changing the training configuration must not have moved them.
- **the structure** - layer list and parameter count, in `test_model.py`.
- **the retune** - that `EarlyStopping` is actually attached with
  `restore_best_weights`, that the epoch ceiling is no longer 5, that seeding
  reproduces an initialisation, and that every fold comes back scored (the
  review's P1 and P2).

These tests import TensorFlow, which `cas12a.data` and `cas12a.features`
deliberately do not; they skip rather than fail where it is absent, so the
TF-free subtree stays testable on its own. `test_metrics.py` holds the scoring
arithmetic and the gate, and needs no TensorFlow at all.
"""

import hashlib
import json

import numpy as np
import pytest

# TensorFlow is optional for this subtree, so the imports below the skip stay
# below it - E402 is not enabled here, and moving them up would defeat it.
pytest.importorskip("tensorflow")

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from cas12a.metrics import CALIBRATION_KEYS, METRIC_KEYS
from cas12a.train import (
    BATCH_SIZE,
    EARLY_STOPPING_MONITOR,
    EPOCHS,
    METRICS_FILENAME,
    N_SPLITS,
    RANDOM_STATE,
    SEED,
    SHUFFLE,
    best_fold_index,
    library_versions,
    make_callbacks,
    make_kfold,
    save_artifacts,
    set_seeds,
    train_5fold,
    training_configuration,
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


def test_the_epoch_ceiling_is_no_longer_five():
    """P2's first half: 5 epochs was the whole of the notebook's training.

    The ceiling is asserted as a lower bound rather than an exact number
    because `EarlyStopping` decides where a fold actually stops - what matters
    is that the loop is no longer cut off before it converges. `batch_size`
    stays 32, untouched, so the per-epoch step count is comparable to the cost
    measurement in `cas12a/docs/cas12a-training-cost.md`.
    """
    assert EPOCHS >= 50
    assert BATCH_SIZE == 32


def test_early_stopping_is_attached_and_restores_the_best_weights():
    """P2's second half, and the one that made reported numbers accidental.

    The notebook imported both callbacks and passed neither to `model.fit`, so
    the weights left in the model were the last epoch's - the review's example
    is fold 4 ending on `val_loss 0.0481` after seeing 0.0445. A callback list
    that is merely built is the same bug, so this reads the objects.
    """
    callbacks = make_callbacks()
    by_type = {type(callback): callback for callback in callbacks}

    stopper = by_type[EarlyStopping]
    assert stopper.restore_best_weights is True
    assert stopper.monitor == EARLY_STOPPING_MONITOR
    assert stopper.patience >= 1

    plateau = by_type[ReduceLROnPlateau]
    assert plateau.monitor == EARLY_STOPPING_MONITOR
    assert plateau.factor < 1.0


def test_op_determinism_stays_off_and_is_recorded_as_such():
    """#92 task 4 fixes seeds and explicitly declines strict determinism.

    The gate is a relative comparison in units of the fold standard deviation,
    which absorbs jitter smaller than itself, and op determinism would only
    slow training down. The JSON says so rather than leaving a reader to infer
    that a seeded run is a bit-identical one.
    """
    config = training_configuration()

    assert config["op_determinism_enabled"] is False
    assert config["seed"] == SEED
    assert config["early_stopping"]["restore_best_weights"] is True


def test_seeding_reproduces_an_initialisation():
    """Same seed, same starting weights - the part of P4 that is testable.

    Weight initialisation draws from the seeded generators, so it *is*
    reproducible; the training that follows is not, which is why this asserts
    on a freshly built model and no test asserts on a trained one.
    """
    from cas12a.model import create_cas12a_model

    set_seeds(SEED)
    first = create_cas12a_model().get_weights()
    set_seeds(SEED)
    second = create_cas12a_model().get_weights()
    set_seeds(SEED + 1)
    other = create_cas12a_model().get_weights()

    assert len(first) == len(second)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, c) for a, c in zip(first, other))


def test_library_versions_records_what_produced_the_numbers():
    versions = library_versions()

    assert {"python", "numpy", "scipy", "scikit-learn", "tensorflow"} <= set(versions)
    assert all(isinstance(value, str) and value for value in versions.values())


def test_fold_indices_match_the_notebook_split():
    folds = _folds()
    assert len(folds) == N_SPLITS

    for fold, (_, val_idx) in enumerate(folds):
        digest = hashlib.sha256(np.asarray(val_idx, dtype=np.int64).tobytes())
        assert tuple(val_idx[:5]) == FOLD_VAL_HEADS[fold], f"fold {fold} head"
        assert digest.hexdigest() == FOLD_VAL_DIGESTS[fold], f"fold {fold} digest"


@pytest.fixture(scope="module")
def tiny_run():
    """The whole loop on 25 synthetic rows, two epochs, no Drive mount.

    Module-scoped because it trains five models and the tests below only read
    what came back. Two epochs is enough to exercise the plumbing - these
    numbers are meaningless as measurements, and nothing here asserts on their
    values, only on their presence and type.
    """
    rows = 25
    rng = np.random.default_rng(0)
    X_seq = rng.random((rows, 34, 4), dtype=np.float32)
    X_feat = rng.random((rows, 354), dtype=np.float32)
    y = rng.random(rows, dtype=np.float32)

    return train_5fold(X_seq, X_feat, y, epochs=2, verbose=0)


def test_every_fold_comes_back_scored(tiny_run):
    """P1 closed: `fold_results` used to be returned empty.

    The notebook initialised the list, appended nothing, and returned it, so no
    number about this model existed anywhere. A caller now gets one dict per
    model, carrying every metric key plus how that fold trained.
    """
    fold_models, fold_results = tiny_run

    assert len(fold_models) == N_SPLITS
    assert len(fold_results) == N_SPLITS

    expected = set(METRIC_KEYS + CALIBRATION_KEYS) | {
        "fold",
        "val_rows",
        "epochs_ran",
        "best_epoch",
        "best_val_loss",
        "final_val_loss",
        "wall_clock_s",
    }
    for index, fold in enumerate(fold_results, start=1):
        assert set(fold) == expected
        assert fold["fold"] == index
        assert fold["val_rows"] == 5


def test_the_reported_epoch_is_the_one_whose_weights_survived(tiny_run):
    """`best_epoch` has to describe the model that is actually returned.

    `restore_best_weights=True` puts back the minimum-`val_loss` epoch, so the
    recorded best epoch is `argmin` of the history and its loss is no worse
    than the final epoch's. Without this, a metrics file could report an epoch
    that no saved weight ever came from - the review's complaint about fold 4.
    """
    _, fold_results = tiny_run

    for fold in fold_results:
        assert 1 <= fold["best_epoch"] <= fold["epochs_ran"]
        assert fold["best_val_loss"] <= fold["final_val_loss"]


class TestArtifacts:
    """#92 task 5: the checkpoint and its numbers, written side by side."""

    def test_best_fold_is_chosen_by_spearman(self):
        results = [
            {"fold": 1, "spearman": 0.70},
            {"fold": 2, "spearman": 0.81},
            {"fold": 3, "spearman": 0.75},
        ]

        assert best_fold_index(results) == 1

    def test_a_collapsed_fold_is_skipped_rather_than_winning(self):
        """NaN must not be picked by a comparison that is False either way."""
        results = [
            {"fold": 1, "spearman": float("nan")},
            {"fold": 2, "spearman": 0.60},
        ]

        assert best_fold_index(results) == 1

    def test_all_folds_collapsed_is_an_error_not_a_saved_model(self):
        results = [{"fold": 1, "spearman": float("nan")}]

        with pytest.raises(ValueError, match="nothing to save"):
            best_fold_index(results)

    def test_the_keras_and_the_json_land_in_the_same_directory(
        self, tiny_run, tmp_path
    ):
        """Which numbers this checkpoint scored, answered next to the file.

        The issue's requirement is not bit-identical retraining - seeds are
        fixed but op determinism is off - but that the file on disk can be
        traced to its metrics. So the JSON names the `.keras` beside it, and
        the fold it names is the one the gate metric picked.
        """
        fold_models, fold_results = tiny_run
        output = tmp_path / "artifacts"

        metrics_path, model_path, _, _ = save_artifacts(
            fold_models, fold_results, output, "synthetic.csv", rows=25
        )

        assert metrics_path.parent == model_path.parent == output
        assert metrics_path.name == METRICS_FILENAME
        assert model_path.exists() and model_path.suffix == ".keras"

        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert record["saved_model"]["filename"] == model_path.name
        best = best_fold_index(fold_results)
        assert record["saved_model"]["fold"] == fold_results[best]["fold"]

    def test_the_json_records_split_configuration_and_versions(
        self, tiny_run, tmp_path
    ):
        """The JSON is the single source of these numbers, so it carries why.

        Per the issue's second comment, a later reader reads this file instead
        of re-reading an executed notebook - which means the split, the
        training configuration and the library versions have to be in it, not
        only the metrics.
        """
        fold_models, fold_results = tiny_run

        metrics_path, _, _, _ = save_artifacts(
            fold_models, fold_results, tmp_path, "synthetic.csv", rows=25
        )
        record = json.loads(metrics_path.read_text(encoding="utf-8"))

        assert record["split"] == {
            "n_splits": N_SPLITS,
            "shuffle": SHUFFLE,
            "random_state": RANDOM_STATE,
        }
        assert record["training"]["op_determinism_enabled"] is False
        assert record["library_versions"]["tensorflow"]
        assert len(record["folds"]) == N_SPLITS
        assert record["gate"]["metric"] == "spearman"
        assert record["dataset"]["rows"] == 25

    def test_the_saved_model_reloads_and_predicts(self, tiny_run, tmp_path):
        """A checkpoint nothing can load is not an artifact.

        P4 was that `fold_models` never reached disk at all; this is the other
        half - that what reaches disk comes back as a model with the two inputs
        the loader produces.
        """
        import tensorflow as tf

        fold_models, fold_results = tiny_run
        _, model_path, _, _ = save_artifacts(
            fold_models, fold_results, tmp_path, "synthetic.csv", rows=25
        )

        reloaded = tf.keras.models.load_model(model_path)
        prediction = reloaded.predict(
            [np.zeros((2, 34, 4), np.float32), np.zeros((2, 354), np.float32)],
            verbose=0,
        )

        assert prediction.shape == (2, 1)
        assert np.all((prediction >= 0) & (prediction <= 1))


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
