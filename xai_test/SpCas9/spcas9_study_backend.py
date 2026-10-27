"""Replaceable training/scoring boundary; framework imports stay inside fitting."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from SpCas9.spcas9_ranking import RANKING_WEIGHT


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

    def __init__(self, *, epochs: int = 50):
        # A smaller value is for bounded disposable synthetic checks only.
        if not 1 <= epochs <= 50:
            raise ValueError("epochs must be between 1 and 50")
        self.epochs = epochs

    def _trainer(self, model):
        return model

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
        initial_digest = hashlib.sha256()
        for weight in model.get_weights():
            initial_digest.update(weight.tobytes())
        trainer = self._trainer(model)
        history = trainer.fit(
            x=training.inputs,
            y=training.labels,
            sample_weight=training.weights,
            validation_data=(validation.inputs, validation.labels, validation.weights),
            epochs=self.epochs,
            batch_size=BATCH_SIZE,
            shuffle=True,
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
            {
                "backend": "keras",
                "device": device_record(),
                "initial_weights_sha256": initial_digest.hexdigest(),
                "optimizer_updates": int(trainer.optimizer.iterations.numpy()),
                **self._objective_record(),
            },
        )

    def _objective_record(self) -> dict:
        return {"optimization_objective": "original binary-masked MSE"}

    def predict(
        self, checkpoint_path: Path, inputs: np.ndarray
    ) -> dict[str, np.ndarray]:
        import keras

        model = keras.models.load_model(checkpoint_path, compile=False)
        return model.predict(inputs, batch_size=1024, verbose=0)


class KerasNggBackend(KerasV2Backend):
    """Fixed #83 candidate; zero weight exists solely for equivalence checks.

    This implements training, not execution readiness. The shared NGG monitor
    and frozen budget scheduler are subsequent tickets (#84/#85).
    """

    def __init__(self, *, auxiliary_weight: float = RANKING_WEIGHT, epochs: int = 50):
        super().__init__(epochs=epochs)
        if auxiliary_weight not in (0.0, RANKING_WEIGHT):
            raise ValueError(
                "only zero-weight verification or the fixed .01 recipe is supported"
            )
        self.auxiliary_weight = auxiliary_weight

    def _trainer(self, model):
        from SpCas9.spcas9_ranking import training_adapter

        return training_adapter(model, auxiliary_weight=self.auxiliary_weight)

    def _objective_record(self) -> dict:
        from SpCas9.spcas9_ranking import RANKING_TEMPERATURE

        return {
            "optimization_objective": "original binary-masked MSE + NGG ranking",
            "ranking": {
                "weight": self.auxiliary_weight,
                "temperature": RANKING_TEMPERATURE,
                "eligibility": "same measured head, both NGG, distinct exact 30mers, unequal labels; each unordered batch pair once",
                "normalization": "per-batch per-head sum(gap * softplus) / sum(gap), empty head zero; fixed nine-head mean",
                "history": "ngg_pairs_<output> and ngg_gap_weight_sum_<output> are epoch sums of batch pair counts and denominators, not an epoch-wide loss denominator",
                "base_rows": "unchanged Keras array shuffle, batch32, including partial final batch; no oversampling",
            },
        }
