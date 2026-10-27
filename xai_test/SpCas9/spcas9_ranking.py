"""Training-only NGG objective. Inference keeps the ordinary functional model."""

from SpCas9.spcas9_data import HEAD_NAMES, output_name

RANKING_WEIGHT = 0.01
RANKING_TEMPERATURE = 0.1


def training_objective(
    inputs, labels, masks, predictions, *, auxiliary_weight=RANKING_WEIGHT
):
    """Sum measured head SSE / actual batch rows; never average by coverage.

    Output-layer names align every head. Boolean selection precedes arithmetic,
    so NaN/Inf missing placeholders cannot poison either values or gradients.
    """
    import tensorflow as tf

    dtype = predictions[output_name(HEAD_NAMES[0])].dtype
    batch_size = tf.shape(inputs)[0]
    tf.debugging.assert_positive(batch_size)
    rows = tf.cast(batch_size, dtype)
    terms = []
    truths, outputs, measured_heads = [], [], []
    for head in HEAD_NAMES:
        name = output_name(head)
        mask = tf.reshape(tf.cast(masks[name], dtype), [-1])
        tf.debugging.assert_equal(tf.shape(mask)[0], batch_size)
        tf.debugging.assert_equal(tf.reduce_all((mask == 0) | (mask == 1)), True)
        measured = mask == 1
        truth = tf.reshape(tf.cast(labels[name], dtype), [-1])
        predicted = tf.reshape(predictions[name], [-1])
        tf.debugging.assert_equal(tf.shape(truth)[0], batch_size)
        tf.debugging.assert_equal(tf.shape(predicted)[0], batch_size)
        truth = tf.where(measured, truth, tf.zeros_like(truth))
        predicted = tf.where(measured, predicted, tf.zeros_like(predicted))
        tf.debugging.assert_all_finite(truth, "non-finite measured label")
        tf.debugging.assert_all_finite(predicted, "non-finite measured prediction")
        terms.append(tf.reduce_sum(tf.square(predicted - truth)) / rows)
        truths.append(truth)
        outputs.append(predicted)
        measured_heads.append(measured)
    mse = tf.add_n(terms)
    # Exact equality over the whole one-hot window, never a protospacer/hash
    # approximation. All pairs use this batch's existing network predictions.
    inputs = tf.convert_to_tensor(inputs)
    different = tf.reduce_any(
        inputs[:, None, :, :] != inputs[None, :, :, :], axis=[2, 3]
    )
    ngg = (inputs[:, 25, 2] == 1) & (inputs[:, 26, 2] == 1)
    indices = tf.range(batch_size)
    unordered = indices[:, None] < indices[None, :]
    measured = tf.stack(measured_heads, axis=1)
    truth = tf.stack(truths, axis=1)
    predicted = tf.stack(outputs, axis=1)
    delta_y = truth[:, None, :] - truth[None, :, :]
    eligible = (
        (unordered & different & ngg[:, None] & ngg[None, :])[:, :, None]
        & measured[:, None, :]
        & measured[None, :, :]
        & (delta_y != 0)
    )
    gaps = tf.where(eligible, tf.abs(delta_y), tf.zeros_like(delta_y))
    delta_p = predicted[:, None, :] - predicted[None, :, :]
    penalties = tf.nn.softplus(-tf.sign(delta_y) * delta_p / RANKING_TEMPERATURE)
    denominators = tf.reduce_sum(gaps, axis=[0, 1])
    by_head = tf.math.divide_no_nan(
        tf.reduce_sum(gaps * penalties, axis=[0, 1]), denominators
    )
    ranking = tf.reduce_sum(by_head) / len(HEAD_NAMES)
    return {
        "loss": mse + tf.cast(auxiliary_weight, dtype) * ranking,
        "masked_mse": mse,
        "ranking": ranking,
        "pair_counts": tf.reduce_sum(tf.cast(eligible, tf.int32), axis=[0, 1]),
        "gap_weight_sums": denominators,
    }


def training_adapter(model, *, auxiliary_weight=RANKING_WEIGHT):
    """A disposable trainer sharing the functional inference model's weights.

    Keras still owns the array adapter, shuffle, batch32, final partial batch,
    forward pass, optimizer, and callbacks. Only loss computation is replaced.
    Save the supplied functional model, never this training-only wrapper.
    """
    import keras
    import tensorflow as tf

    from SpCas9.spcas9_train import compile_model

    if auxiliary_weight not in (0.0, RANKING_WEIGHT):
        raise ValueError(
            "only zero-weight verification or the fixed .01 recipe is supported"
        )

    class NggTrainer(keras.Model):
        def __init__(self):
            super().__init__(inputs=model.inputs, outputs=model.output)
            self.objective_trackers = {
                "masked_mse": keras.metrics.Mean(name="masked_mse"),
                "ngg_ranking": keras.metrics.Mean(name="ngg_ranking"),
            }
            for head in HEAD_NAMES:
                for statistic in ("pairs", "gap_weight_sum"):
                    name = f"ngg_{statistic}_{output_name(head)}"
                    self.objective_trackers[name] = keras.metrics.Sum(name=name)

        def compute_loss(
            self, x=None, y=None, y_pred=None, sample_weight=None, training=True
        ):
            result = training_objective(
                x, y, sample_weight, y_pred, auxiliary_weight=auxiliary_weight
            )
            rows = tf.cast(tf.shape(x)[0], result["loss"].dtype)
            self.objective_trackers["masked_mse"].update_state(
                result["masked_mse"], sample_weight=rows
            )
            self.objective_trackers["ngg_ranking"].update_state(
                result["ranking"], sample_weight=rows
            )
            for index, head in enumerate(HEAD_NAMES):
                self.objective_trackers[f"ngg_pairs_{output_name(head)}"].update_state(
                    result["pair_counts"][index]
                )
                self.objective_trackers[
                    f"ngg_gap_weight_sum_{output_name(head)}"
                ].update_state(result["gap_weight_sums"][index])
            return result["loss"]

        def compute_metrics(self, x, y, y_pred, sample_weight=None):
            # Weighted MAE also needs finite missing placeholders before Keras
            # arithmetic; a zero sample weight alone cannot remove a NaN.
            clean_y = {
                name: tf.where(
                    tf.reshape(tf.cast(sample_weight[name], tf.bool), tf.shape(values)),
                    values,
                    tf.zeros_like(values),
                )
                for name, values in y.items()
            }
            return super().compute_metrics(
                x, clean_y, y_pred, sample_weight=sample_weight
            )

    trainer = NggTrainer()
    compile_model(trainer)
    return trainer
