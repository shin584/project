"""Replaceable training/scoring boundary; framework imports stay inside fitting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class TrainingRows:
    """Only sequence inputs, named labels and weights cross into training."""

    inputs: np.ndarray
    labels: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]


@dataclass
class FitResult:
    history: dict[str, list[float]]
    validation_predictions: dict[str, np.ndarray]
    environment: dict


class TrainingBackend(Protocol):
    def fit(
        self,
        training: TrainingRows,
        validation: TrainingRows,
        *,
        seed: int,
        checkpoint_path: Path,
    ) -> FitResult: ...

    def predict(
        self, checkpoint_path: Path, inputs: np.ndarray
    ) -> dict[str, np.ndarray]: ...


class KerasV2Backend:
    """The historical v2 model/optimizer with explicit common-loss monitoring."""

    def fit(
        self,
        training: TrainingRows,
        validation: TrainingRows,
        *,
        seed: int,
        checkpoint_path: Path,
    ) -> FitResult:
        import keras

        from SpCas9.spcas9_train import (
            BATCH_SIZE,
            EPOCHS,
            REFERENCE_LOSS,
            build_model,
            compile_model,
            device_record,
            reference_validation_loss,
            training_callbacks,
        )

        class ReferenceValidation(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                outputs = self.model.predict(
                    validation.inputs, batch_size=1024, verbose=0
                )
                logs[REFERENCE_LOSS] = reference_validation_loss(
                    validation.labels,
                    validation.weights,
                    outputs,
                )

        keras.utils.set_random_seed(seed)
        model = build_model()
        compile_model(model)
        history = model.fit(
            x=training.inputs,
            y=training.labels,
            sample_weight=training.weights,
            validation_data=(validation.inputs, validation.labels, validation.weights),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                ReferenceValidation(),
                *training_callbacks(monitor=REFERENCE_LOSS),
            ],
            verbose=1,
        )
        # EarlyStopping restored the common-criterion best weights on train end.
        model.save(checkpoint_path)
        values = {
            name: [float(v) for v in series] for name, series in history.history.items()
        }
        values["recipe_optimization_loss"] = values.pop("loss")
        values["val_recipe_optimization_loss"] = values.pop("val_loss")
        return FitResult(
            values,
            model.predict(validation.inputs, batch_size=1024, verbose=0),
            {"backend": "keras", "device": device_record()},
        )

    def predict(
        self, checkpoint_path: Path, inputs: np.ndarray
    ) -> dict[str, np.ndarray]:
        import keras

        model = keras.models.load_model(checkpoint_path, compile=False)
        return model.predict(inputs, batch_size=1024, verbose=0)
