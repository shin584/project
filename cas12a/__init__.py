"""Cas12a cleavage-efficiency model: features, data loading, score units, training.

`features` and `data` are pure NumPy/pandas; `model` and `train` are where
TensorFlow enters, so the first two stay importable without it. The dependency
runs one way - `train` -> `model` -> `features` - and nothing here imports
`e_system`.

The same split holds for the two training entry points added since: `train`
fits five folds on a `KFold` it builds itself, and `cnn_single` fits one model
on membership read back from a saved partition (#116). `cnn_manifest` is that
reader and stays NumPy/pandas, so "which rows, and what are their labels" is
answerable without a deep-learning framework; `cnn_single` is the TensorFlow
half and imports its recipe from `train` rather than restating it.
"""
