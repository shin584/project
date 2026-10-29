"""Cas12a cleavage-efficiency model: features, data loading, score units, training.

`features` and `data` are pure NumPy/pandas; `model` and `train` are where
TensorFlow enters, so the first two stay importable without it. The dependency
runs one way - `train` -> `model` -> `features` - and nothing here imports
`e_system`.
"""
