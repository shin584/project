"""Structure parity of the ported network with the notebook's (#93 task 5).

Weights cannot be compared - they are random at build time and training is
stochastic - so what is pinned is the shape of the thing: which layers exist,
what each one outputs, and how many parameters it holds. Those three were
checked equal, layer for layer, against `create_cas12a_model` as it stood in
`cas12a/model/Cas12a_model_final.ipynb` before the port, and the golden table
below is that comparison frozen.

`#91` removed the notebook's dead `bean_input` branch, which never executed, so
it changed none of these numbers (see #93's first comment).
"""

import pytest

# TensorFlow is optional for this subtree, so the imports below the skip stay
# below it - E402 is not enabled here, and moving them up would defeat it.
pytest.importorskip("tensorflow")

from cas12a.features import FEATURE_DIM, SEQUENCE_LENGTH
from cas12a.model import (
    CONV_FILTERS,
    CONV_KERNEL_SIZES,
    LEARNING_RATE,
    create_cas12a_model,
)

#: `(layer class, output shape, parameter count)` in `model.layers` order, as
#: the notebook's model reports them. The order is Keras's own topological one,
#: which interleaves the two inputs' branches - that interleaving is part of
#: what is being pinned, so the list is kept exactly as emitted.
GOLDEN_LAYERS = (
    ("InputLayer", (None, 34, 4), 0),
    ("Conv1D", (None, 34, 64), 1088),
    ("Conv1D", (None, 34, 64), 2112),
    ("Conv1D", (None, 34, 64), 3904),
    ("BatchNormalization", (None, 34, 64), 256),
    ("BatchNormalization", (None, 34, 64), 256),
    ("BatchNormalization", (None, 34, 64), 256),
    ("LeakyReLU", (None, 34, 64), 0),
    ("LeakyReLU", (None, 34, 64), 0),
    ("LeakyReLU", (None, 34, 64), 0),
    ("Concatenate", (None, 34, 192), 0),
    ("Dropout", (None, 34, 192), 0),
    ("Bidirectional", (None, 34, 64), 57600),
    ("InputLayer", (None, 354), 0),
    ("Dropout", (None, 34, 64), 0),
    ("Dense", (None, 64), 22720),
    ("GlobalAveragePooling1D", (None, 64), 0),
    ("Dropout", (None, 64), 0),
    ("Concatenate", (None, 128), 0),
    ("Dense", (None, 64), 8256),
    ("Dense", (None, 1), 65),
)

GOLDEN_PARAM_COUNT = 96513


@pytest.fixture(scope="module")
def model():
    """One model for the module - building it costs a second or two."""
    return create_cas12a_model()


def test_layer_table_matches_the_notebook(model):
    observed = tuple(
        (layer.__class__.__name__, tuple(layer.output.shape), layer.count_params())
        for layer in model.layers
    )
    assert observed == GOLDEN_LAYERS


def test_parameter_count_matches_the_notebook(model):
    assert model.count_params() == GOLDEN_PARAM_COUNT


def test_inputs_match_what_the_loader_returns(model):
    """The two input shapes are the loader's two arrays, by construction.

    `cas12a.data.load_cas12a_dataset` returns `(n, 34, 4)` and `(n, 354)`, and
    both shapes come from `cas12a.features` constants rather than from literals
    repeated here - so this asserts the wiring, not the numbers.
    """
    seq_input, feat_input = model.inputs

    assert seq_input.shape[1:] == (SEQUENCE_LENGTH, 4)
    assert feat_input.shape[1:] == (FEATURE_DIM,)
    assert [tensor.name for tensor in model.inputs] == ["Input_Seq", "Input_Feat"]


def test_output_is_a_bounded_single_score(model):
    """One sigmoid output, which is what `cas12a/units.py` promises.

    A score leaving this model is already inside `[0, 1]`, unlike SaCas9's
    linear head - that asymmetry is the reason `units.clip_score` exists.
    """
    assert model.output.shape[1:] == (1,)
    assert model.layers[-1].activation.__name__ == "sigmoid"


def test_three_convolutions_read_the_window_at_three_widths(model):
    """4 bp, 8 bp and 15 bp kernels, 64 filters each, over the 34 bp input."""
    conv_layers = [
        layer for layer in model.layers if layer.__class__.__name__ == "Conv1D"
    ]

    assert [layer.kernel_size[0] for layer in conv_layers] == list(CONV_KERNEL_SIZES)
    assert {layer.filters for layer in conv_layers} == {CONV_FILTERS}
    assert {layer.padding for layer in conv_layers} == {"same"}


def test_compiled_with_the_notebook_s_optimiser_and_loss(model):
    """Adam at 5e-4, MSE, MAE as the only metric.

    MAE is not an evaluation metric in any useful sense - it is reported
    per-epoch and nothing collects it, which is the review's P1. Pinned here so
    that #92 adding real metrics shows up as an intended change.
    """
    assert model.optimizer.__class__.__name__ == "Adam"
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(LEARNING_RATE)
    assert model.loss == "mse"
