"""Scoring one fold's predictions, and the adoption gate (#92).

This is the module the review's P1 was missing: the notebook imported
`mean_absolute_error`, `r2_score` and `pearsonr` without calling any of them,
and never imported Spearman at all, so no number about this model existed
(`cas12a/docs/cas12a-model-review.md` §3).

Pure NumPy/SciPy/scikit-learn - no TensorFlow - so the metric arithmetic is
testable on arrays alone, the way `cas12a/features.py` and `cas12a/data.py`
are. The dependency runs one way: `train` -> `metrics`, never back.

**Spearman is the gate metric and MAE plus calibration are reported beside
it.** Rank correlation is what the field reports and what §2's ridge baseline
is quoted in, but the app compares *absolute* scores across Cas types, so a
model whose ranking is right and whose level is shifted would break the
cross-type comparison while scoring well on Spearman alone. `CALIBRATION_KEYS`
is that second axis.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#: RidgeCV's 5-fold Spearman on the same features and the same split
#: (`cas12a/docs/cas12a-model-review.md` §2). The gate, not an aspiration: a
#: CNN+BiLSTM that does not clear a linear model does not justify its cost.
BASELINE_SPEARMAN = 0.738

#: DeepCpf1's reported Spearman on this dataset. A **reference line, not a
#: gate** - Kim 2018 split and preprocessed differently, so falling short of it
#: cannot be attributed to the architecture rather than to the conditions.
REFERENCE_SPEARMAN = 0.77

#: The metric the gate reads.
GATE_KEY = "spearman"

#: Keys every fold reports, in the order they are printed.
METRIC_KEYS = (
    "spearman",
    "pearson",
    "r2",
    "mae",
    "mse",
)

#: The absolute-level companions to the rank metrics above. Perfect
#: calibration is `slope=1`, `intercept=0`, `mean_bias=0`.
CALIBRATION_KEYS = (
    "calibration_slope",
    "calibration_intercept",
    "mean_bias",
)


def _correlation(statistic, y_true, y_pred):
    """A correlation, or NaN where it is undefined rather than a warning.

    Both correlations divide by a standard deviation, so a fold that predicted
    one constant value has no correlation to report. That is a real outcome
    (a collapsed model) and it is recorded as NaN instead of crashing the run
    that would otherwise have told us about it.
    """
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(statistic(y_true, y_pred)[0])


def fold_metrics(y_true, y_pred):
    """Score one fold's validation predictions.

    Returns a flat dict over `METRIC_KEYS + CALIBRATION_KEYS`. `y_pred` may
    carry Keras's trailing singleton axis - `(n, 1)` as well as `(n,)` - and is
    flattened here so the caller does not have to remember to.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_pred {y_pred.shape} disagree in length"
        )

    slope, intercept = _calibration_fit(y_true, y_pred)

    return {
        "spearman": _correlation(spearmanr, y_true, y_pred),
        "pearson": _correlation(pearsonr, y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_bias": float(np.mean(y_pred) - np.mean(y_true)),
    }


def _calibration_fit(y_true, y_pred):
    """Least-squares `y_true ~ slope * y_pred + intercept`.

    Regressing truth *on* the prediction is the direction that answers "when
    this model says 0.8, what does the data do?", which is the question the
    app's cross-Cas-type comparison asks. A constant prediction has no slope,
    so both come back NaN.
    """
    if np.std(y_pred) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(y_pred, y_true, 1)
    return float(slope), float(intercept)


def aggregate(folds):
    """Mean and population std per metric, over a list of `fold_metrics` dicts.

    `ddof=0`: the five folds are the whole of what the gate compares against,
    not a sample drawn from a larger pool of folds. NaN from a collapsed fold
    propagates on purpose - a summary that silently dropped it would read as a
    clean result.
    """
    if not folds:
        raise ValueError("no folds to aggregate")

    keys = METRIC_KEYS + CALIBRATION_KEYS
    return {
        key: {
            "mean": float(np.mean([fold[key] for fold in folds])),
            "std": float(np.std([fold[key] for fold in folds], ddof=0)),
        }
        for key in keys
    }


def gate_verdict(summary, baseline=BASELINE_SPEARMAN):
    """Decide adoption: mean Spearman must clear `baseline` by at least 1 std.

    The issue's wording is "exceeds the ridge baseline by at least the fold
    standard deviation", so the margin is `mean - baseline` and it is compared
    to `std`. A run whose folds disagree widely has to win by more, which is
    the point: five folds spread 0.05 apart do not establish a 0.01 win.

    NaN anywhere fails - `>=` is False against NaN - so a collapsed fold cannot
    be read as a pass.
    """
    mean = summary[GATE_KEY]["mean"]
    std = summary[GATE_KEY]["std"]
    margin = mean - baseline

    return {
        "metric": GATE_KEY,
        "baseline": baseline,
        "mean": mean,
        "std": std,
        "margin": margin,
        "passed": bool(margin >= std),
        "reference_only": {"deepcpf1_spearman": REFERENCE_SPEARMAN},
    }


def format_summary(folds, summary, verdict):
    """The run's one-screen report, as lines.

    Returned rather than printed so a test can read it and `train.py` owns the
    output stream.
    """
    lines = ["", "--- Per-fold metrics ---"]
    header = "fold  " + "  ".join(f"{key:>10}" for key in METRIC_KEYS)
    lines.append(header)
    for index, fold in enumerate(folds, start=1):
        row = "  ".join(f"{fold[key]:>10.4f}" for key in METRIC_KEYS)
        lines.append(f"{index:<6}{row}")

    lines.append("")
    lines.append("--- Mean +/- std over folds ---")
    for key in METRIC_KEYS + CALIBRATION_KEYS:
        stats = summary[key]
        lines.append(f"{key:>22}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    lines.append("")
    lines.append("--- Adoption gate ---")
    lines.append(
        f"Spearman {verdict['mean']:.4f} +/- {verdict['std']:.4f} vs ridge "
        f"baseline {verdict['baseline']:.4f}: margin {verdict['margin']:+.4f}, "
        f"needs >= {verdict['std']:.4f} (1 fold std)"
    )
    lines.append("PASSED - adoptable" if verdict["passed"] else "NOT PASSED")
    lines.append(
        f"(DeepCpf1 {REFERENCE_SPEARMAN} is a reference line, not a gate - "
        "different split and preprocessing.)"
    )
    return lines
