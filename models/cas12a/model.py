"""The Cas12a cleavage-efficiency network.

Ported from `cas12a/model/Cas12a_model_final.ipynb` by #93, layer for layer.
Nothing here is new and nothing here is tuned: the architecture, the
initialisers' defaults, the optimiser and the loss are the notebook's, so that a
change in reported performance after the port can only come from the port
itself and not from a quiet retune. The settings the review flagged
(`cas12a/docs/cas12a-model-review.md` §3) are #92's to change.

TensorFlow lives here and in `cas12a/train.py` only. `cas12a/features.py` and
`cas12a/data.py` stay pure NumPy/pandas, and the dependency runs one way:
`model` -> `features`, never back. `e_system` is not imported from anywhere in
this package.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LeakyReLU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from cas12a.features import FEATURE_DIM, SEQUENCE_LENGTH

#: Adam's learning rate, as the notebook set it.
LEARNING_RATE = 0.0005
#: Widths of the three parallel convolutions over the one-hot sequence. Three
#: receptive fields - 4 bp, 8 bp, 15 bp - read the 34 bp window at once.
CONV_KERNEL_SIZES = (4, 8, 15)
CONV_FILTERS = 64
LSTM_UNITS = 32
LEAKY_RELU_SLOPE = 0.1


def create_cas12a_model():
    """Build and compile the two-input Cas12a regressor.

    Takes `[X_seq, X_feat]` - `(n, 34, 4)` one-hot and `(n, 354)` GC +
    dinucleotide features, the two arrays `cas12a.data.load_cas12a_dataset`
    returns - and predicts a single 0-1 efficiency through a sigmoid, which is
    the output-unit contract in `cas12a/units.py`.

    The sequence branch is three parallel `Conv1D`s, concatenated, then a
    BiLSTM pooled over time; the feature branch is one `Dense`. They meet in a
    final `Dense` pair. Returns a model compiled with Adam/MSE and MAE as its
    only metric - the review's P1 (no evaluation metric at all) is #92's.
    """
    seq_input = Input(shape=(SEQUENCE_LENGTH, 4), name="Input_Seq")

    conv_branches = []
    for kernel_size in CONV_KERNEL_SIZES:
        branch = Conv1D(CONV_FILTERS, kernel_size=kernel_size, padding="same")(
            seq_input
        )
        branch = BatchNormalization()(branch)
        branch = LeakyReLU(negative_slope=LEAKY_RELU_SLOPE)(branch)
        conv_branches.append(branch)

    x = tf.keras.layers.Concatenate()(conv_branches)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    x = GlobalAveragePooling1D()(x)

    feat_input = Input(shape=(FEATURE_DIM,), name="Input_Feat")
    f = Dense(64, activation="relu")(feat_input)
    f = Dropout(0.3)(f)

    x = tf.keras.layers.Concatenate()([x, f])
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[seq_input, feat_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"]
    )
    return model
