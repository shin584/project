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


def _metric_fn(metric: str) -> MetricFn:
    if metric not in METRIC_FUNCS:
        raise ValueError(
            f"Unknown metric: {metric!r}. Choose from {sorted(METRIC_FUNCS)}."
        )
    return METRIC_FUNCS[metric]


def _interval(deltas: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = 1 - ci
    bounds = np.asarray(np.quantile(deltas, [alpha / 2, 1 - alpha / 2]))
    return float(bounds[0]), float(bounds[1])


def joint_clustered_bootstrap_ci(
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray], tuple[float, float] | None],
    *,
    n_resamples: int,
    maximum_attempts: int,
    ci: float,
    seed: int,
) -> dict:
    """Two jointly valid statistics from one shared cluster/multiplicity draw.

    The caller receives row indices once per attempt and applies them to every
    compared array. A cluster's observations always travel together. Undefined
    joint statistics consume an attempt, never a valid-draw slot. A short quota
    reports unavailable intervals rather than intervals from a smaller sample.
    """
    clusters = np.asarray(clusters)
    if clusters.ndim != 1 or not 0 < ci < 1:
        raise ValueError("one-dimensional clusters and 0 < ci < 1 required")
    if n_resamples <= 0 or maximum_attempts <= 0:
        raise ValueError("positive draw quota and attempt limit required")
    _, inverse = np.unique(clusters, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == c) for c in range(len(np.unique(clusters)))]
    rng = np.random.default_rng(seed)
    valid: list[tuple[float, float]] = []
    attempts = 0
    while rows_of and len(valid) < n_resamples and attempts < maximum_attempts:
        attempts += 1
        drawn = rng.integers(0, len(rows_of), size=len(rows_of))
        idx = np.concatenate([rows_of[c] for c in drawn])
        values = statistic(idx)
        if values is not None and np.isfinite(values).all():
            valid.append(values)
    complete = len(valid) == n_resamples
    return {
        "status": "available" if complete else "unavailable",
        "valid_draws": len(valid),
        "attempts": attempts,
        "invalid_draws": attempts - len(valid),
        "reason": None
        if complete
        else (
            "no eligible clusters"
            if not rows_of
            else "valid joint draw quota unmet at maximum attempts; required summary undefined in rejected draws"
        ),
        "intervals": [list(_interval(np.asarray(valid)[:, i], ci)) for i in range(2)]
        if complete
        else None,
    }


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

    Each draw resamples **rows**, which assumes the rows are independent. Where
    several rows are replicate measurements of one unit - the same sequence
    scored more than once - use `clustered_paired_bootstrap_delta_ci` instead,
    or this interval will be narrower than the data supports.
    """
    metric_fn = _metric_fn(metric)
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

    return _interval(deltas, ci)


def clustered_paired_bootstrap_delta_ci(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    clusters: np.ndarray,
    metric: str,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """`paired_bootstrap_delta_ci` resampling whole clusters, not rows.

    `clusters` labels each row with the unit that is actually independent -
    for a sequence model scored on replicate rows, the sequence. Each draw
    takes as many cluster labels as there are, with replacement, and keeps
    every row of a drawn cluster, so a unit measured several times is drawn
    once rather than counted as several independent observations. Drawn
    samples therefore vary in row count, which is what a cluster bootstrap
    does: the number of independent units is what is held fixed.

    With one row per cluster this is exactly the row bootstrap.
    """
    metric_fn = _metric_fn(metric)
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    clusters = np.asarray(clusters)
    if len(clusters) != len(y_true):
        raise ValueError(
            f"got {len(clusters)} cluster labels for {len(y_true)} rows; "
            "every row needs one"
        )

    _, inverse = np.unique(clusters, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == c) for c in range(inverse.max() + 1)]
    n_clusters = len(rows_of)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([rows_of[c] for c in drawn])
        deltas[i] = metric_fn(y_true[idx], preds_a[idx]) - metric_fn(
            y_true[idx], preds_b[idx]
        )

    return _interval(deltas, ci)
