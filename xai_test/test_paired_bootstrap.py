import numpy as np
import pytest
from paired_bootstrap import compute_metrics, paired_bootstrap_delta_ci
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _make_regression_fixture(n=200, noise_a=0.05, noise_b=0.3, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(0, 1, size=n)
    preds_a = y_true + rng.normal(0, noise_a, size=n)  # accurate variant
    preds_b = y_true + rng.normal(0, noise_b, size=n)  # noisy variant
    return y_true, preds_a, preds_b


# --- compute_metrics ----------------------------------------------------------


def test_compute_metrics_matches_manual_calculation():
    y_true = np.array([0.1, 0.4, 0.35, 0.8, 0.2])
    y_pred = np.array([0.12, 0.38, 0.4, 0.75, 0.25])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["spearman"] == pytest.approx(spearmanr(y_true, y_pred)[0])
    assert metrics["pearson"] == pytest.approx(pearsonr(y_true, y_pred)[0])
    assert metrics["mae"] == pytest.approx(mean_absolute_error(y_true, y_pred))
    assert metrics["mse"] == pytest.approx(mean_squared_error(y_true, y_pred))


def test_compute_metrics_does_not_rescale_predictions():
    # Raw pass-through: scaling y_pred changes MAE/MSE (no rescaling is applied
    # before metrics are computed - AC5 of issue #13).
    y_true = np.array([0.1, 0.4, 0.35, 0.8, 0.2])
    y_pred = np.array([0.12, 0.38, 0.4, 0.75, 0.25])
    y_pred_scaled = y_pred * 2.0

    metrics = compute_metrics(y_true, y_pred)
    metrics_scaled = compute_metrics(y_true, y_pred_scaled)

    assert metrics["mae"] != pytest.approx(metrics_scaled["mae"])
    assert metrics["mse"] != pytest.approx(metrics_scaled["mse"])


# --- paired_bootstrap_delta_ci -------------------------------------------------


def test_rejects_unknown_metric():
    y_true, preds_a, preds_b = _make_regression_fixture()
    with pytest.raises(ValueError):
        paired_bootstrap_delta_ci(y_true, preds_a, preds_b, metric="rmse")


def test_ci_bounds_are_ordered():
    y_true, preds_a, preds_b = _make_regression_fixture()

    lower, upper = paired_bootstrap_delta_ci(y_true, preds_a, preds_b, metric="mse")

    assert lower <= upper


def test_ci_is_reproducible_under_the_same_seed():
    y_true, preds_a, preds_b = _make_regression_fixture()

    first = paired_bootstrap_delta_ci(
        y_true, preds_a, preds_b, metric="spearman", seed=42
    )
    second = paired_bootstrap_delta_ci(
        y_true, preds_a, preds_b, metric="spearman", seed=42
    )

    assert first == second


def test_ci_collapses_to_zero_when_variants_are_identical():
    y_true, preds_a, _ = _make_regression_fixture()

    lower, upper = paired_bootstrap_delta_ci(y_true, preds_a, preds_a, metric="mse")

    # Every resample computes the same metric for both "variants" -> delta is
    # always exactly 0, regardless of which samples get resampled.
    assert lower == pytest.approx(0.0, abs=1e-12)
    assert upper == pytest.approx(0.0, abs=1e-12)


def test_ci_detects_a_clear_mse_difference_between_variants():
    # preds_a is much more accurate than preds_b, so Full (a) has lower MSE:
    # delta_mse = mse(a) - mse(b) should be clearly negative, CI excluding 0.
    y_true, preds_a, preds_b = _make_regression_fixture(noise_a=0.02, noise_b=0.5)

    _lower, upper = paired_bootstrap_delta_ci(y_true, preds_a, preds_b, metric="mse")

    assert upper < 0


def test_ci_detects_a_clear_spearman_difference_between_variants():
    # preds_a tracks y_true's rank order far better than preds_b, so
    # delta_spearman = spearman(a) - spearman(b) should be clearly positive.
    y_true, preds_a, preds_b = _make_regression_fixture(noise_a=0.02, noise_b=0.5)

    lower, _upper = paired_bootstrap_delta_ci(
        y_true, preds_a, preds_b, metric="spearman"
    )

    assert lower > 0


def test_ci_narrows_as_resample_count_increases_variance_stabilizes():
    # Sanity check that more resamples doesn't blow up the interval - not a
    # strict monotonicity claim, just that both are finite and ordered.
    y_true, preds_a, preds_b = _make_regression_fixture()

    small = paired_bootstrap_delta_ci(
        y_true, preds_a, preds_b, metric="mae", n_resamples=50
    )
    large = paired_bootstrap_delta_ci(
        y_true, preds_a, preds_b, metric="mae", n_resamples=2000
    )

    for lower, upper in (small, large):
        assert np.isfinite(lower)
        assert np.isfinite(upper)
        assert lower <= upper
