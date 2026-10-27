"""Synthetic checks of the #83 training path; never fit a published split."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from SpCas9.spcas9_data import HEAD_NAMES, one_hot, output_name


@pytest.fixture(scope="module", autouse=True)
def synthetic_budget():
    """Opt-in actual-backend checks share one durable cumulative ledger.

    An unfinished prior session blocks reruns. Reserved updates count even on
    failure. #85 must import training_seconds_charged, including failed checks.
    """
    ledger_name = os.environ.get("SPCAS9_SYNTHETIC_LEDGER")
    if not ledger_name:
        pytest.skip("actual-backend checks require an explicit SPCAS9_SYNTHETIC_LEDGER")
    assert os.environ.get("SPCAS9_SYNTHETIC_SESSION"), (
        "use python -m SpCas9.spcas9_ranking_checks for the external timeout"
    )
    path = Path(ledger_name)
    record = json.loads(path.read_text(encoding="utf-8"))
    session = record["sessions"][-1]
    assert session["id"] == os.environ["SPCAS9_SYNTHETIC_SESSION"]
    assert session["status"] == "running"
    assert record["optimizer_updates_charged"] <= 10
    assert record["training_seconds_charged"] < 300

    def persist():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def reserve_updates(updates):
        assert record["optimizer_updates_charged"] + updates <= 10
        record["optimizer_updates_charged"] += updates
        session["optimizer_updates_reserved"] += updates
        persist()

    yield reserve_updates


def fixture_batch():
    sequences = [
        "AAAA" + "A" * 20 + "AGG" + "AAA",
        "AAAA" + "C" * 20 + "TGG" + "AAA",
        "AAAA" + "G" * 20 + "AGC" + "AAA",
    ]
    labels, masks = {}, {}
    for index, head in enumerate(HEAD_NAMES):
        name = output_name(head)
        labels[name] = np.array([0.0, 0.2 + index / 20, -0.05], np.float32)
        masks[name] = np.array([1, index % 2, 1], np.float32)
    return one_hot(sequences), labels, masks


def test_zero_auxiliary_matches_keras_scalar_and_every_trainable_gradient():
    import keras
    import tensorflow as tf

    from SpCas9.spcas9_ranking import training_objective
    from SpCas9.spcas9_train import build_model, compile_model

    keras.utils.set_random_seed(42)
    model = build_model()
    compile_model(model)
    inputs, labels, masks = fixture_batch()
    # Both objectives use the very same forward pass: identical BatchNorm,
    # Dropout, dtype and stochastic state, with no optimizer update.
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(inputs, training=True)
        reference = model.compute_loss(
            x=inputs, y=labels, y_pred=predictions, sample_weight=masks
        )
        actual = training_objective(
            inputs, labels, masks, predictions, auxiliary_weight=0.0
        )["loss"]
        combined = training_objective(inputs, labels, masks, predictions)["loss"]
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
    expected_gradients = tape.gradient(reference, model.trainable_weights)
    gradients = tape.gradient(actual, model.trainable_weights)
    assert len(gradients) > 0
    for expected, gradient in zip(expected_gradients, gradients, strict=True):
        assert expected is not None and gradient is not None
        np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-6)
    assert np.isfinite(combined)
    for gradient in tape.gradient(combined, model.trainable_weights):
        assert gradient is not None and np.isfinite(gradient).all()


def test_ranking_excludes_missing_non_ngg_duplicates_and_ties_without_a_gap_cutoff():
    import tensorflow as tf

    from SpCas9.spcas9_ranking import training_objective

    sequences = [
        "AAAA" + "A" * 20 + "AGGAAA",
        "AAAA" + "C" * 20 + "TGGAAA",
        "AAAA" + "G" * 20 + "AGCAAA",
        "AAAA" + "A" * 20 + "AGGAAA",  # repeated exact input
        "AAAA" + "T" * 20 + "AGGAAA",
        "CAAA" + "T" * 20 + "AGGAAA",
    ]
    names = [output_name(head) for head in HEAD_NAMES]
    labels = {name: tf.constant([np.nan] * 6, tf.float32) for name in names}
    masks = {name: tf.zeros(6) for name in names}
    predictions = {name: tf.zeros((6, 1)) for name in reversed(names)}
    labels[names[0]] = tf.constant([0, 0.2, 0.3, 0.4, 0.2, 0.200001])
    masks[names[0]] = tf.ones(6)
    labels[names[1]] = tf.constant([0, 0.2, np.nan, np.inf, -np.inf, np.nan])
    masks[names[1]] = tf.constant([1, 1, 0, 0, 0, 0])

    result = training_objective(
        one_hot(sequences), labels, masks, predictions, auxiliary_weight=0.01
    )

    np.testing.assert_array_equal(result["pair_counts"], [8, 1, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(
        result["gap_weight_sums"][:2], [1.200002, 0.2], atol=1e-7
    )
    assert float(result["ranking"]) == pytest.approx(2 * np.log(2) / 9)
    assert float(result["masked_mse"]) == pytest.approx(0.06833340000016667)
    assert float(result["loss"]) == pytest.approx(0.06987372706807766)

    changed = {
        name: tf.where(masks[name] == 1, value, 12345.0)
        for name, value in labels.items()
    }
    unchanged = training_objective(
        one_hot(sequences), changed, masks, predictions, auxiliary_weight=0.01
    )
    assert float(unchanged["loss"]) == float(result["loss"])


def test_ranking_uses_gap_weights_and_rewards_correct_order():
    import tensorflow as tf

    from SpCas9.spcas9_ranking import training_objective

    inputs, _, _ = fixture_batch()
    # Turn the third row into NGG; all three exact windows still differ.
    inputs[2, 26] = inputs[0, 26]
    names = [output_name(head) for head in HEAD_NAMES]
    labels = {name: tf.constant([0.0, 0.2, 0.5]) for name in names}
    masks = {name: tf.zeros(3) for name in names}
    masks[names[0]] = tf.ones(3)
    predictions = {name: tf.constant([[0.0], [0.1], [-0.2]]) for name in names}
    result = training_objective(
        inputs, labels, masks, predictions, auxiliary_weight=0.01
    )
    # Worked three-pair example: gap weights .2, .5, .3; logits -1, 2, 3.
    assert float(result["ranking"]) == pytest.approx(0.22674361649969482)

    masks[names[0]] = tf.constant([1, 1, 0])
    correct = training_objective(
        inputs, labels, masks, predictions, auxiliary_weight=0.01
    )
    reversed_predictions = {name: -value for name, value in predictions.items()}
    reversed_result = training_objective(
        inputs, labels, masks, reversed_predictions, auxiliary_weight=0.01
    )
    assert float(correct["ranking"]) < float(reversed_result["ranking"])


def test_backend_preserves_fit_order_partial_batch_and_plain_checkpoint(
    tmp_path, synthetic_budget
):
    import keras

    from SpCas9.spcas9_study_backend import (
        KerasNggBackend,
        KerasV2Backend,
        TrainingRows,
    )

    inputs, labels, masks = fixture_batch()
    # 33 base rows: batch32 plus a genuine final batch of one (no pairs).
    row_ids = np.arange(33) % 3
    training = TrainingRows(
        inputs[row_ids],
        {k: v[row_ids] for k, v in labels.items()},
        {k: v[row_ids] for k, v in masks.items()},
    )
    validation = TrainingRows(inputs, labels, masks)
    reference_path, zero_path, ranked_path = (
        tmp_path / f"{name}.keras" for name in ("reference", "zero", "ranked")
    )

    # Reserve before entering fit; even failed checks consume their allowance.
    synthetic_budget(6)
    reference = KerasV2Backend(epochs=1).fit(
        training, validation, seed=42, checkpoint_path=reference_path
    )
    zero = KerasNggBackend(auxiliary_weight=0.0, epochs=1).fit(
        training, validation, seed=42, checkpoint_path=zero_path
    )
    assert zero.history["recipe_optimization_loss"] == pytest.approx(
        reference.history["recipe_optimization_loss"], rel=1e-5, abs=1e-6
    )
    reference_model = keras.models.load_model(reference_path)
    zero_model = keras.models.load_model(zero_path)
    for left, right in zip(reference_model.weights, zero_model.weights, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)

    # Missing placeholders must also be safe in the actual fit/weighted MAE.
    training.labels = {
        k: np.where(training.weights[k] == 1, v, np.nan)
        for k, v in training.labels.items()
    }
    backend = KerasNggBackend(epochs=1)
    ranked = backend.fit(training, validation, seed=42, checkpoint_path=ranked_path)
    assert all(np.isfinite(series).all() for series in ranked.history.values())
    assert ranked.environment["optimizer_updates"] == 2
    assert (
        ranked.environment["initial_weights_sha256"]
        == zero.environment["initial_weights_sha256"]
    )
    assert (
        ranked.environment["initial_weights_sha256"]
        == reference.environment["initial_weights_sha256"]
    )
    assert ranked.environment["ranking"]["weight"] == 0.01
    assert ranked.environment["ranking"]["temperature"] == 0.1
    assert ranked.history[f"ngg_pairs_{output_name(HEAD_NAMES[0])}"][0] > 0
    assert ranked.history[f"ngg_gap_weight_sum_{output_name(HEAD_NAMES[0])}"][0] > 0
    # The ordinary loader receives no custom objects, including when compiled.
    reloaded = keras.models.load_model(ranked_path)
    assert type(reloaded).__name__ == "Functional"
    after = backend.predict(ranked_path, inputs)
    assert set(after) == {output_name(head) for head in HEAD_NAMES}
    for name, before in ranked.validation_predictions.items():
        assert np.isfinite(before).all()
        np.testing.assert_allclose(after[name], before, rtol=1e-5, atol=1e-6)
