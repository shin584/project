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
- `full_metrics` / `error_metrics`: MAE/MSE and correlations under #32's
  minimum-n rule, and the historical n/MAE/MSE-only variant. Full metrics
  explain unavailable values and keep reports strict JSON.
- `head_metrics`: the two put together - a head's columns pulled out of a
  predictions frame and scored.
- `population_metrics` / `equal_head_ngg_summary`: measured NGG/non-NGG
  groups and the equal-weight mean over all nine required NGG heads (#82).

Pure arithmetic over arrays and frames: no model, no file, no dataset.
"""

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from SpCas9.spcas9_data import (
    HEAD_NAMES,
    SEQUENCE_COLUMN,
    is_ngg,
    output_name,
    protospacers,
)

# Correlations on fewer measured pairs are too unstable to report.
MIN_N_FOR_SPEARMAN = 10


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
    """Measured-pair metrics with the existing minimum-n rule and reasons.

    Invalid measured values invalidate the group; they are never dropped to
    manufacture a finite result. The head mask must be applied by the caller.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("metrics require equally sized one-dimensional arrays")
    n = len(y_true)
    values: dict[str, float | None] = dict.fromkeys(
        ("mae", "mse", "spearman", "pearson")
    )
    reasons = {}
    if not n:
        reasons = dict.fromkeys(values, "no measured rows")
    elif not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        reasons = dict.fromkeys(values, "non-finite measured label or prediction")
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            residual = y_pred - y_true
            values["mae"] = float(np.mean(np.abs(residual)))
            values["mse"] = float(np.mean(residual**2))
        if n < MIN_N_FOR_SPEARMAN:
            reason = f"requires at least {MIN_N_FOR_SPEARMAN} measured rows"
        elif np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
            reason = "constant measured labels or predictions"
        else:
            reason = None
            values["spearman"] = float(spearmanr(y_true, y_pred)[0])
            values["pearson"] = float(pearsonr(y_true, y_pred)[0])
        if reason:
            reasons.update(spearman=reason, pearson=reason)
        for key, value in values.items():
            if _finite_or_none(value) is None and key not in reasons:
                reasons[key] = "non-finite computed metric"
    return {
        "n": n,
        **{key: _finite_or_none(value) for key, value in values.items()},
        "unavailable_reasons": reasons,
    }


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """n / MAE / MSE only, for groups where a correlation means nothing."""
    metrics = full_metrics(y_true, y_pred)
    return {key: metrics[key] for key in ("n", "mae", "mse")}


def head_metrics(frame: pd.DataFrame, head: str, metric=full_metrics) -> dict:
    """`metric` over `head`'s rows of a predictions frame.

    The pairing every scoring step repeats: pull a head's two columns out of a
    predictions frame, then score them. `metric` is `error_metrics` for a
    group where a correlation means nothing.
    """
    return metric(*head_arrays(frame, head))


def target_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Rows and distinct inputs, without deduplicating repeated observations."""
    return {
        "n_rows": len(frame),
        "unique_30mers": int(frame[SEQUENCE_COLUMN].nunique()),
        "unique_protospacers": int(protospacers(frame).nunique()),
    }


def population_metrics(frame: pd.DataFrame) -> dict:
    """NGG and its complement, each scored on every head's measured rows.

    Pooled n counts measured head/row pairs; n_rows counts distinct eligible
    rows. Pooled correlations are computed over the concatenated pairs, never
    from a mean of head correlations.
    """
    mask = is_ngg(frame)
    populations = {}
    for name, selected in (("ngg", mask), ("non_ngg", ~mask)):
        rows = frame[selected]
        heads, labels, predictions = {}, [], []
        any_measured = np.zeros(len(rows), dtype=bool)
        for head in HEAD_NAMES:
            measured = rows[true_column(head)].notna().to_numpy()
            eligible = rows[measured]
            y_true, y_pred = head_arrays(eligible, head)
            heads[head] = {**full_metrics(y_true, y_pred), **target_counts(eligible)}
            labels.append(y_true)
            predictions.append(y_pred)
            any_measured |= measured
        populations[name] = {
            "rows": len(rows),
            "heads": heads,
            "pooled": {
                **full_metrics(np.concatenate(labels), np.concatenate(predictions)),
                **target_counts(rows[any_measured]),
            },
        }
    return populations


def equal_head_ngg_summary(populations: dict) -> dict:
    """Arithmetic means across all nine heads; any missing metric blocks its mean."""
    heads = populations["ngg"]["heads"]
    summary: dict = {"required_heads": list(HEAD_NAMES), "unavailable_reasons": {}}
    for metric in ("mae", "spearman"):
        unavailable = {
            head: heads[head]["unavailable_reasons"][metric]
            for head in HEAD_NAMES
            if heads[head][metric] is None
        }
        summary[metric] = (
            None
            if unavailable
            else math.fsum(heads[head][metric] / len(HEAD_NAMES) for head in HEAD_NAMES)
        )
        if unavailable:
            summary["unavailable_reasons"][metric] = unavailable
    return summary
