"""Per-head metric vocabulary shared by every 9-Hydra scoring step (issue #48).

The pieces the 9-Hydra v1 leakage audit, the v2 scoring run and the v1 <-> v2
comparison all have to agree on, so that two reports of "Spearman for
`SpCas9`" mean the same arithmetic:

- `true_column` / `pred_column` / `head_arrays`: the `<head>_true` /
  `<head>_pred` column naming a predictions frame uses, and the pair of
  float arrays a metric is computed from.
- `predictions_by_head`: the model's outputs re-keyed from output **layer**
  name to head name, via the shared loader's `output_name`. Every 9-Hydra
  checkpoint carries the nine heads in an order that disagrees with the
  spreadsheet's on `eSpCas9(1.1)` / `SpCas9-HF.1`, so predictions are never
  matched to labels by position.
- `full_metrics` / `error_metrics`: `leakage_check.group_metrics` (#32) plus
  Pearson under the same minimum-n rule, and the n/MAE/MSE-only variant for
  groups where a correlation means nothing. Both return `None` rather than
  NaN for an undefined figure, so a report stays strict JSON.
- `head_metrics`: the two put together - a head's columns pulled out of a
  predictions frame and scored.

Pure arithmetic over arrays and frames: no model, no file, no dataset.
"""

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from leakage_check import MIN_N_FOR_SPEARMAN, group_metrics
from scipy.stats import pearsonr

from SpCas9.spcas9_data import HEAD_NAMES, output_name


def true_column(head: str) -> str:
    return f"{head}_true"


def pred_column(head: str) -> str:
    return f"{head}_pred"


def head_arrays(frame: pd.DataFrame, head: str) -> tuple[np.ndarray, np.ndarray]:
    """`head`'s labels and predictions from a predictions frame, as floats."""
    return (
        frame[true_column(head)].to_numpy(dtype=np.float64),
        frame[pred_column(head)].to_numpy(dtype=np.float64),
    )


def predictions_by_head(outputs: Mapping[str, object]) -> dict[str, np.ndarray]:
    """The model's outputs re-keyed by head name, flattened to 1-D.

    `outputs` is keyed by output layer name - what a dict-output model (v2)
    hands back from `predict`, and what a list-output model (v1) becomes once
    zipped against its own `output_names`.
    """
    by_output = {
        name: np.asarray(out, dtype=np.float64).reshape(-1)
        for name, out in outputs.items()
    }
    missing = [head for head in HEAD_NAMES if output_name(head) not in by_output]
    if missing:
        raise KeyError(f"checkpoint has no output for heads {missing}")
    return {head: by_output[output_name(head)] for head in HEAD_NAMES}


def _finite_or_none(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """`group_metrics` (#32) plus Pearson, under the same minimum-n rule."""
    metrics = group_metrics(y_true, y_pred)
    pearson = (
        float(pearsonr(y_true, y_pred)[0])
        if len(y_true) >= MIN_N_FOR_SPEARMAN
        else None
    )
    metrics["pearson"] = pearson
    return {key: _finite_or_none(v) if key != "n" else v for key, v in metrics.items()}


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """n / MAE / MSE only, for groups where a correlation means nothing."""
    metrics = group_metrics(y_true, y_pred)
    return {key: metrics[key] for key in ("n", "mae", "mse")}


def head_metrics(frame: pd.DataFrame, head: str, metric=full_metrics) -> dict:
    """`metric` over `head`'s rows of a predictions frame.

    The pairing every scoring step repeats: pull a head's two columns out of a
    predictions frame, then score them. `metric` is `error_metrics` for a
    group where a correlation means nothing.
    """
    return metric(*head_arrays(frame, head))
