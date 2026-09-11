"""Paired bootstrap significance testing for Model B ablation (Step 4-1, issue #13).

Each bootstrap draw resamples the SAME index set (with replacement) for
y_true and every compared variant's predictions together, so the comparison
stays paired - it isolates the metric delta itself rather than treating the
two variants as independent samples with their own sampling noise.
"""

from collections.abc import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

MetricFn = Callable[[np.ndarray, np.ndarray], float]

METRIC_FUNCS: dict[str, MetricFn] = {
    "spearman": lambda y_true, y_pred: spearmanr(y_true, y_pred)[0],
    "pearson": lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Spearman/Pearson/MAE/MSE on raw values - no cross-model rescaling."""
    return {name: float(fn(y_true, y_pred)) for name, fn in METRIC_FUNCS.items()}


def paired_bootstrap_delta_ci(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    metric: str,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """CI for `metric(preds_a) - metric(preds_b)` under paired resampling.

    `metric` must be a key of `METRIC_FUNCS`. Returns `(lower, upper)` for the
    `ci` central interval (default 95%) over `n_resamples` bootstrap draws.
    """
    if metric not in METRIC_FUNCS:
        raise ValueError(
            f"Unknown metric: {metric!r}. Choose from {sorted(METRIC_FUNCS)}."
        )

    metric_fn = METRIC_FUNCS[metric]
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = metric_fn(y_true[idx], preds_a[idx]) - metric_fn(
            y_true[idx], preds_b[idx]
        )

    alpha = 1 - ci
    lower, upper = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)
