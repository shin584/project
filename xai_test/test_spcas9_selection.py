"""The existing per-run backend boundary, replaying epochs without fitting."""

import numpy as np
import pytest

from SpCas9.spcas9_data import HEAD_NAMES, one_hot, output_name
from SpCas9.spcas9_study_backend import KerasNggBackend, KerasV2Backend, TrainingRows

MAE = "val_equal_head_ngg_mae"
SPEARMAN = "val_mean_ngg_spearman"


@pytest.fixture
def validation_rows():
    sequences = ["AAAA" + base * 20 + "AGGAAA" for base in "ACGTA"] * 2
    # Repeated input and a measured zero are real observations. Non-NGG must
    # not affect selection; one differently masked row has a NaN placeholder.
    sequences += ["CCCC" + "A" * 20 + "AGGAAA", "AAAA" + "A" * 20 + "AGCAAA"]
    labels = {
        output_name(h): np.array([0, 1, 2, 3, 4] * 2 + [np.nan, -100.0])
        for h in HEAD_NAMES
    }
    weights = {name: np.array([1] * 10 + [0, 1.0]) for name in labels}
    labels[output_name(HEAD_NAMES[1])][10] = 2
    weights[output_name(HEAD_NAMES[1])][10] = 1
    return TrainingRows(one_hot(sequences), labels, weights)


def replay_framework(
    monkeypatch,
    *,
    epochs=4,
    corrupt=None,
    monitor_fault=None,
    real_inference=False,
    capture_real=False,
):
    """Replace only external framework fit/inference; real weights serialize.

    No optimizer update, stochastic training pass or actual-split fit occurs.
    Keras itself dispatches the production callbacks and stopping/LR policy.
    """
    import keras

    observed = []

    def predict(model, inputs, **kwargs):
        epoch = round(
            float(model.get_layer(output_name(HEAD_NAMES[0])).bias.numpy()[0])
        )
        ordered = [5, 6, 7, 8, 9] if epoch in (2, 3) else [9, 8, 7, 6, 5]
        if epoch >= 4:
            ordered = [v + 1 for v in ordered]
        outputs = {
            output_name(h): np.array(
                ordered * 2 + [7 + (epoch >= 4), 10000.0], dtype=float
            )[:, None]
            for h in reversed(HEAD_NAMES)
        }
        if corrupt:
            corrupt(epoch, outputs)
        return outputs

    def fit(model, **kwargs):
        callbacks = keras.callbacks.CallbackList(
            kwargs["callbacks"], model=model, add_history=True
        )
        callbacks.on_train_begin()
        for epoch in range(epochs):
            logs = {"loss": 10.0 - epoch, "val_loss": 10.0 - epoch}
            callbacks.on_epoch_begin(epoch, logs)
            # Epoch identity is a real saved weight, independent of monitors.
            for head in HEAD_NAMES:
                model.get_layer(output_name(head)).bias.assign([epoch + 1])
            if monitor_fault:
                monitor_fault(callbacks, epoch, logs)
            else:
                callbacks.on_epoch_end(epoch, logs)
            observed.append(
                {name: values[-1] for name, values in model.history.history.items()}
            )
            if capture_real:
                observed[-1]["actual_predictions"] = {
                    name: value.numpy().copy()
                    for name, value in model(
                        kwargs["validation_data"][0], training=False
                    ).items()
                }
            if model.stop_training:
                break
        callbacks.on_train_end()
        return model.history

    monkeypatch.setattr(keras.Model, "fit", fit)
    if not real_inference:
        monkeypatch.setattr(keras.Model, "predict", predict)
    return observed


@pytest.mark.parametrize("backend_class", [KerasV2Backend, KerasNggBackend])
def test_run_selects_spearman_tie_then_earlier_epoch_and_reloads(
    tmp_path, monkeypatch, validation_rows, backend_class
):
    observed = replay_framework(monkeypatch, capture_real=True)
    path = tmp_path / "selected.keras"
    backend = backend_class()
    result = backend.fit(
        validation_rows, validation_rows, seed=42, checkpoint_path=path
    )

    assert result.selection["best_epoch"] == 2
    assert result.selection["history"][MAE] == pytest.approx([5, 5, 5, 6])
    assert result.selection["history"][SPEARMAN] == pytest.approx([-1, 1, 1, -1])
    assert result.selection["save_reload_parity"]["passed"] is True
    assert result.environment["optimizer_updates"] == 0
    assert all(MAE in logs and SPEARMAN in logs for logs in observed)
    assert result.history[MAE] == pytest.approx([5, 5, 5, 6])
    import keras

    saved = keras.models.load_model(path)
    assert saved.get_layer(output_name(HEAD_NAMES[0])).bias.numpy()[0] == 2
    # Use the real reloaded graph to reproduce all nine predictions captured
    # at the Spearman-selected epoch, rather than the later tied epoch.
    actual = saved(validation_rows.inputs, training=False)
    for name, values in actual.items():
        np.testing.assert_allclose(
            values, observed[1]["actual_predictions"][name], rtol=1e-5, atol=1e-6
        )
        assert not np.allclose(values, observed[2]["actual_predictions"][name])
    reloaded = backend.predict(path, validation_rows.inputs)
    for name, values in result.validation_predictions.items():
        np.testing.assert_array_equal(reloaded[name], values)
        np.testing.assert_array_equal(values[:5, 0], [5, 6, 7, 8, 9])


@pytest.mark.parametrize("backend_class", [KerasV2Backend, KerasNggBackend])
def test_spearman_tie_break_does_not_reset_mae_patience_or_lr(
    tmp_path, monkeypatch, validation_rows, backend_class
):
    observed = replay_framework(monkeypatch, epochs=10)
    result = backend_class().fit(
        validation_rows,
        validation_rows,
        seed=42,
        checkpoint_path=tmp_path / "early.keras",
    )
    assert len(observed) == 6  # MAE last improved at epoch1; patience stays five.
    assert observed[4]["learning_rate"] == pytest.approx(0.0002)
    assert result.selection["best_epoch"] == 2


@pytest.mark.parametrize(
    "failure", ["nan", "inf", "missing_head", "constant", "short", "mask", "empty"]
)
def test_run_refuses_unavailable_required_monitors(
    tmp_path, monkeypatch, validation_rows, failure
):
    def corrupt(epoch, outputs):
        name = output_name(HEAD_NAMES[0])
        if failure == "missing_head":
            del outputs[name]
        elif failure == "constant":
            outputs[name][:] = 7
        elif failure in ("nan", "inf"):
            outputs[name][0] = float(failure)

    if failure == "short":
        validation_rows.weights[output_name(HEAD_NAMES[0])][0] = 0
    if failure == "mask":
        validation_rows.weights[output_name(HEAD_NAMES[0])][0] = 0.5
    if failure == "empty":
        validation_rows.weights[output_name(HEAD_NAMES[0])][:] = 0
    replay_framework(monkeypatch, corrupt=corrupt)
    path = tmp_path / "invalid.keras"
    with pytest.raises((ValueError, KeyError)):
        KerasV2Backend().fit(
            validation_rows, validation_rows, seed=42, checkpoint_path=path
        )
    assert not path.exists()


@pytest.mark.parametrize("failure", ["missing", "nonfinite", "stale"])
def test_consumers_refuse_missing_nonfinite_or_stale_current_epoch_values(
    tmp_path, monkeypatch, validation_rows, failure
):
    previous = {}

    def fault(callbacks, epoch, logs):
        if failure == "stale" and epoch == 1:
            logs.update(previous)
        else:
            callbacks.callbacks[0].on_epoch_end(epoch, logs)
        previous.update(logs)
        if failure == "missing":
            logs.pop(SPEARMAN)
        if failure == "nonfinite":
            logs[MAE] = np.inf
        for callback in callbacks.callbacks[1:]:
            callback.on_epoch_end(epoch, logs)

    replay_framework(monkeypatch, monitor_fault=fault)
    path = tmp_path / "invalid.keras"
    with pytest.raises(ValueError, match="NGG monitor"):
        KerasV2Backend().fit(
            validation_rows, validation_rows, seed=42, checkpoint_path=path
        )
    assert not path.exists()


def test_actual_inference_predictions_survive_selector_restoration_and_plain_reload(
    tmp_path, monkeypatch, validation_rows
):
    # Real graph, real forward passes, real serialization. Only fit dispatch
    # is replaced with epoch weight assignments: no gradients or updates.
    replay_framework(monkeypatch, real_inference=True)
    path = tmp_path / "actual.keras"
    backend = KerasV2Backend()
    result = backend.fit(
        validation_rows, validation_rows, seed=42, checkpoint_path=path
    )
    after = backend.predict(path, validation_rows.inputs)
    assert result.environment["optimizer_updates"] == 0
    assert result.selection["save_reload_parity"]["passed"]
    for name, selected in result.validation_predictions.items():
        np.testing.assert_allclose(after[name], selected, rtol=1e-5, atol=1e-6)
