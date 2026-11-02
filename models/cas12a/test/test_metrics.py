"""The scoring and the gate (#92).

`cas12a/metrics.py` is the module the review's P1 was missing, so these tests
hold two things: that each metric computes what its name says on arrays whose
answer is known by hand, and that the adoption gate cannot be passed by
accident - not by a fold that collapsed to a constant, and not by a mean that
beats the ridge baseline by less than the folds disagree among themselves.

No TensorFlow here, deliberately: metric arithmetic is array arithmetic, and
this file runs in the TF-free part of the subtree alongside `test_features.py`
and `test_data.py`.
"""

import math

import numpy as np
import pytest

from cas12a.metrics import (
    BASELINE_SPEARMAN,
    CALIBRATION_KEYS,
    GATE_KEY,
    METRIC_KEYS,
    REFERENCE_SPEARMAN,
    aggregate,
    fold_metrics,
    format_summary,
    gate_verdict,
)

#: A fold's worth of targets with no ties, so the rank metrics are unambiguous.
TRUTH = np.array([0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95])


def _summary(spearman_per_fold):
    """An `aggregate` output carrying the given per-fold Spearman values.

    The other metrics are filled with the same numbers because the gate reads
    Spearman alone; this keeps the gate tests about the gate.
    """
    folds = [
        {key: value for key in METRIC_KEYS + CALIBRATION_KEYS}
        for value in spearman_per_fold
    ]
    return aggregate(folds)


def test_baseline_is_the_ridge_number_from_the_review():
    """0.738 is RidgeCV's Spearman on the same features and the same split.

    Pinned because the gate is *relative* to it: if this constant drifts, every
    adoption decision made against it silently changes meaning.
    """
    assert BASELINE_SPEARMAN == 0.738
    assert REFERENCE_SPEARMAN == 0.77
    assert GATE_KEY == "spearman"


class TestFoldMetrics:
    def test_a_perfect_prediction_scores_perfectly(self):
        scored = fold_metrics(TRUTH, TRUTH)

        assert scored["spearman"] == pytest.approx(1.0)
        assert scored["pearson"] == pytest.approx(1.0)
        assert scored["r2"] == pytest.approx(1.0)
        assert scored["mae"] == pytest.approx(0.0)
        assert scored["mse"] == pytest.approx(0.0)
        assert scored["calibration_slope"] == pytest.approx(1.0)
        assert scored["calibration_intercept"] == pytest.approx(0.0, abs=1e-12)
        assert scored["mean_bias"] == pytest.approx(0.0)

    def test_rank_and_linear_correlation_come_apart(self):
        """Spearman 1 with Pearson below it, on a monotone curve.

        This is why the gate reads Spearman and the review quotes it: a
        prediction can order every row correctly while its values bend away
        from the truth. The pair is reported together so that case is visible
        rather than hidden behind one number.
        """
        scored = fold_metrics(TRUTH, TRUTH**3)

        assert scored["spearman"] == pytest.approx(1.0)
        assert scored["pearson"] < 0.95

    def test_calibration_catches_a_shifted_prediction_that_ranks_perfectly(self):
        """+0.1 on every row: Spearman 1, and the shift shows up anyway.

        The app compares absolute scores across Cas types, so a model that is
        uniformly optimistic breaks cross-type comparison while scoring a
        perfect rank correlation. `mean_bias` and the intercept are what make
        that readable.
        """
        scored = fold_metrics(TRUTH, TRUTH + 0.1)

        assert scored["spearman"] == pytest.approx(1.0)
        assert scored["mean_bias"] == pytest.approx(0.1)
        assert scored["calibration_slope"] == pytest.approx(1.0)
        assert scored["calibration_intercept"] == pytest.approx(-0.1)
        assert scored["mae"] == pytest.approx(0.1)

    def test_an_inverted_prediction_scores_minus_one(self):
        scored = fold_metrics(TRUTH, -TRUTH)

        assert scored["spearman"] == pytest.approx(-1.0)
        assert scored["pearson"] == pytest.approx(-1.0)
        assert scored["r2"] < 0

    def test_a_collapsed_fold_reports_nan_instead_of_raising(self):
        """A model that predicts one value has no correlation to report.

        It is a real outcome and the run that found it should finish and say
        so, so the correlations and the calibration fit come back NaN while MAE
        - which is still defined - does not.
        """
        scored = fold_metrics(TRUTH, np.full_like(TRUTH, 0.5))

        assert math.isnan(scored["spearman"])
        assert math.isnan(scored["pearson"])
        assert math.isnan(scored["calibration_slope"])
        assert math.isnan(scored["calibration_intercept"])
        assert scored["mae"] == pytest.approx(np.mean(np.abs(TRUTH - 0.5)))

    def test_keras_column_predictions_are_accepted(self):
        """`model.predict` returns `(n, 1)`; `(n,)` must score the same."""
        flat = fold_metrics(TRUTH, TRUTH * 0.9)
        column = fold_metrics(TRUTH, (TRUTH * 0.9).reshape(-1, 1))

        assert flat == column

    def test_length_mismatch_is_an_error_not_a_broadcast(self):
        """Silent broadcasting here would invent a metric out of nothing."""
        with pytest.raises(ValueError, match="disagree in length"):
            fold_metrics(TRUTH, TRUTH[:-1])

    def test_every_declared_key_is_present(self):
        scored = fold_metrics(TRUTH, TRUTH * 0.5)

        assert set(scored) == set(METRIC_KEYS + CALIBRATION_KEYS)


class TestAggregate:
    def test_mean_and_population_std_over_folds(self):
        """`ddof=0`: the five folds are the whole comparison, not a sample."""
        summary = _summary([0.70, 0.74, 0.78])

        assert summary["spearman"]["mean"] == pytest.approx(0.74)
        assert summary["spearman"]["std"] == pytest.approx(np.std([0.70, 0.74, 0.78]))

    def test_a_nan_fold_propagates_rather_than_being_dropped(self):
        """A summary that hid a collapsed fold would read as a clean result."""
        summary = _summary([0.70, float("nan"), 0.78])

        assert math.isnan(summary["spearman"]["mean"])

    def test_no_folds_is_an_error(self):
        with pytest.raises(ValueError, match="no folds"):
            aggregate([])


class TestGate:
    def test_a_win_wider_than_the_fold_spread_passes(self):
        verdict = gate_verdict(_summary([0.80, 0.81, 0.82, 0.81, 0.80]))

        assert verdict["margin"] > verdict["std"]
        assert verdict["passed"] is True

    def test_a_win_narrower_than_the_fold_spread_does_not(self):
        """Mean 0.75 beats 0.738, and the folds disagree by more than that.

        This is the case the gate exists for: five folds spread far apart do
        not establish a 0.012 win over a linear model.
        """
        verdict = gate_verdict(_summary([0.68, 0.72, 0.75, 0.78, 0.82]))

        assert verdict["mean"] > BASELINE_SPEARMAN
        assert verdict["passed"] is False

    def test_losing_to_the_baseline_does_not_pass(self):
        verdict = gate_verdict(_summary([0.70, 0.70, 0.70, 0.70, 0.70]))

        assert verdict["margin"] < 0
        assert verdict["passed"] is False

    def test_a_nan_fold_cannot_pass(self):
        """NaN compares False, and that is the outcome we want pinned."""
        verdict = gate_verdict(_summary([0.95, float("nan"), 0.95]))

        assert verdict["passed"] is False

    def test_the_verdict_is_json_serialisable_booleans(self):
        """`np.bool_` would survive `json.dumps` differently; force `bool`."""
        verdict = gate_verdict(_summary([0.90, 0.90, 0.90]))

        assert isinstance(verdict["passed"], bool)

    def test_deepcpf1_is_recorded_as_reference_only(self):
        """0.77 must not become a second gate - different split, different prep."""
        verdict = gate_verdict(_summary([0.90, 0.90, 0.90]))

        assert verdict["baseline"] == BASELINE_SPEARMAN
        assert verdict["reference_only"]["deepcpf1_spearman"] == REFERENCE_SPEARMAN


class TestFormatSummary:
    def _rendered(self, spearman_per_fold):
        folds = [
            {key: value for key in METRIC_KEYS + CALIBRATION_KEYS}
            for value in spearman_per_fold
        ]
        summary = aggregate(folds)
        return "\n".join(format_summary(folds, summary, gate_verdict(summary)))

    def test_a_pass_says_so_and_a_failure_says_so(self):
        assert "PASSED - adoptable" in self._rendered([0.90, 0.90, 0.90])
        assert "NOT PASSED" in self._rendered([0.50, 0.50, 0.50])

    def test_one_row_per_fold_plus_every_metric(self):
        rendered = self._rendered([0.80, 0.81, 0.82])

        for key in METRIC_KEYS + CALIBRATION_KEYS:
            assert key in rendered
        assert "DeepCpf1" in rendered
