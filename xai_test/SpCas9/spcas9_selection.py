"""Shared NGG validation and authoritative checkpoint selection (issue #84).

Framework imports remain lazy. Neither candidate loss nor Test is an input.
"""

import numpy as np
import pandas as pd

from SpCas9.spcas9_data import HEAD_NAMES, SEQUENCE_COLUMN, is_ngg, output_name
from SpCas9.spcas9_metrics import equal_head_ngg_summary, full_metrics

NGG_MAE = "val_equal_head_ngg_mae"
NGG_SPEARMAN = "val_mean_ngg_spearman"


def validation_ngg_mask(inputs: np.ndarray) -> np.ndarray:
    """Recover the exact sequence population from validated one-hot inputs."""
    inputs = np.asarray(inputs)
    if (
        inputs.ndim != 3
        or inputs.shape[1:] != (30, 4)
        or not len(inputs)
        or not np.isin(inputs, [0, 1]).all()
        or not (inputs.sum(axis=2) == 1).all()
    ):
        raise ValueError("NGG validation requires nonempty one-hot 30bp sequences")
    bases = np.asarray(list("ACGT"))[inputs.argmax(axis=2)]
    sequences = ["".join(row) for row in bases]
    return is_ngg(pd.DataFrame({SEQUENCE_COLUMN: sequences})).to_numpy(dtype=bool)


def ngg_summary(labels, weights, predictions, ngg: np.ndarray) -> dict:
    """The #82 measured-head arithmetic, with undefined monitors refused."""
    heads = {}
    for head in HEAD_NAMES:
        name = output_name(head)
        truth = np.asarray(labels[name], dtype=float).reshape(-1)
        mask = np.asarray(weights[name], dtype=float).reshape(-1)
        predicted = np.asarray(predictions[name], dtype=float).reshape(-1)
        if any(len(values) != len(ngg) for values in (truth, mask, predicted)):
            raise ValueError(f"{head}: NGG validation row mismatch")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{head}: NGG validation requires binary head masks")
        measured = ngg & (mask == 1)
        heads[head] = full_metrics(truth[measured], predicted[measured])
    summary = equal_head_ngg_summary({"ngg": {"heads": heads}})
    if summary["unavailable_reasons"]:
        raise ValueError(
            f"unavailable required NGG monitor: {summary['unavailable_reasons']}"
        )
    return summary


def ngg_stopping(history: dict[str, list[float]]) -> dict:
    """One-based epoch, exact MAE/Spearman ties, and the complete monitor history."""
    series = {}
    for name in (NGG_MAE, NGG_SPEARMAN):
        values = np.asarray(history.get(name, []), dtype=float)
        if (
            values.ndim != 1
            or not 1 <= len(values) <= 50
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"missing or non-finite required NGG monitor history: {name}"
            )
        series[name] = values
    mae, spearman = series[NGG_MAE], series[NGG_SPEARMAN]
    if len(mae) != len(spearman):
        raise ValueError("NGG monitor history length mismatch")
    best = min(range(len(mae)), key=lambda i: (mae[i], -spearman[i], i))
    return {
        "monitor": NGG_MAE,
        "tie_break": ["higher NGG Spearman", "earlier epoch"],
        "epochs_run": len(mae),
        "best_epoch": best + 1,
        "best_ngg_mae": float(mae[best]),
        "best_ngg_spearman": float(spearman[best]),
        "history": {name: values.tolist() for name, values in series.items()},
        "restore_best_weights": True,
        "weights_owner": "NGG selector",
    }


def ngg_callbacks(validation):
    """Publish current predictions before selection, stopping and LR consumers.

    Epoch-start clearing plus an epoch token prevents a reused logs dictionary
    from turning an omitted producer into a stale, apparently valid monitor.
    """
    import keras

    from SpCas9.spcas9_train import training_callbacks

    ngg = validation_ngg_mask(validation.inputs)

    class ValidationMonitor(keras.callbacks.Callback):
        current_epoch = None
        predictions = None

        def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = None
            if logs is not None:
                for name in (NGG_MAE, NGG_SPEARMAN):
                    logs.pop(name, None)

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                raise ValueError("missing epoch logs for required NGG monitors")
            predictions = self.model.predict(
                validation.inputs, batch_size=1024, verbose=0
            )
            summary = ngg_summary(
                validation.labels, validation.weights, predictions, ngg
            )
            logs[NGG_MAE], logs[NGG_SPEARMAN] = summary["mae"], summary["spearman"]
            self.predictions = predictions
            self.current_epoch = epoch

    producer = ValidationMonitor()

    class CheckpointSelector(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.history = {NGG_MAE: [], NGG_SPEARMAN: []}
            self.best_weights = None
            self.predictions = None
            self.best_key = None

        def on_epoch_end(self, epoch, logs=None):
            if producer.current_epoch != epoch or epoch != len(self.history[NGG_MAE]):
                raise ValueError(
                    "missing current-epoch NGG monitors; stale values refused"
                )
            if logs is None or any(
                name not in logs or not np.isfinite(logs[name])
                for name in (NGG_MAE, NGG_SPEARMAN)
            ):
                raise ValueError("missing or non-finite required NGG monitor")
            mae, spearman = float(logs[NGG_MAE]), float(logs[NGG_SPEARMAN])
            self.history[NGG_MAE].append(mae)
            self.history[NGG_SPEARMAN].append(spearman)
            key = (mae, -spearman, epoch)
            if self.best_key is None or key < self.best_key:
                self.best_key = key
                self.best_weights = self.model.get_weights()
                self.predictions = {
                    name: np.array(values, copy=True)
                    for name, values in producer.predictions.items()
                }

        def on_train_end(self, logs=None):
            if self.best_weights is None:
                raise ValueError("no epoch with valid NGG selection evidence")
            self.model.set_weights(self.best_weights)

    selector = CheckpointSelector()
    callbacks = [
        producer,
        selector,
        *training_callbacks(monitor=NGG_MAE, restore_best_weights=False),
    ]
    return callbacks, selector
