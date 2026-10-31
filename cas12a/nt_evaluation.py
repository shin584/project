"""NT final metrics and report serialization without a training runtime."""

import csv
from datetime import datetime, timezone

import numpy as np
from scipy.stats import pearsonr, spearmanr

from cas12a import nt_storage as storage
from cas12a.nt_data import canonical_bytes

COMPARISON_POLICY = (
    "The historical CNN five-fold mean Spearman of approximately 0.751 in "
    "cas12a/artifacts/cas12a_5fold_metrics.json is not directly comparable to "
    "the NT-Cas12a Testset holdout: evaluation populations and procedures differ. "
    "Historical CNN checkpoint training populations overlap the NT holdout; "
    "using an existing CNN checkpoint is not a leakage-free comparison. "
    "This report makes no NT-versus-CNN superiority claim and applies no CNN, "
    "Ridge or train-mean superiority gate."
)


def _diagnostics(values):
    finite = bool(np.isfinite(values).all())
    with np.errstate(over="ignore", invalid="ignore"):
        variance = float(np.var(values)) if finite else None
    return {
        "all_finite": finite,
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(values))),
        "minimum": float(values.min()) if finite else None,
        "maximum": float(values.max()) if finite else None,
        "variance": variance
        if variance is not None and np.isfinite(variance)
        else None,
        "out_of_range_count": int(np.count_nonzero((values < 0) | (values > 1))),
        "constant": bool(finite and values.min() == values.max()),
    }


def _metrics(labels, predictions):
    finite = bool(np.isfinite(labels).all() and np.isfinite(predictions).all())
    result = {}
    reason = None
    if not finite:
        reason = "nonfinite labels or predictions"
    elif len(labels) < 2:
        reason = "fewer than two targets"
    elif labels.min() == labels.max():
        reason = "constant labels"
    elif predictions.min() == predictions.max():
        reason = "constant predictions"
    for name, statistic in (("spearman", spearmanr), ("pearson", pearsonr)):
        value = (
            float(statistic(labels, predictions).statistic) if reason is None else None
        )
        undefined = reason
        if value is not None and not np.isfinite(value):
            value, undefined = None, "numerically undefined correlation"
        result[name] = value
        result[f"{name}_undefined_reason"] = undefined
    with np.errstate(over="ignore", invalid="ignore"):
        errors = predictions - labels
        for name, value in (
            ("mae", np.mean(np.abs(errors))),
            ("mse", np.mean(errors**2)),
        ):
            result[name] = float(value) if finite and np.isfinite(value) else None
    return result


def prediction_metrics(labels, raw_predictions, baseline_value):
    """Score the two fixed views and frozen baseline on identical ordered targets.

    Invalid numerical results remain JSON-safe failed evidence. Finite constant
    predictions retain their actual errors even though correlations are undefined.
    """
    labels = np.asarray(labels, dtype=np.float64)
    raw = np.asarray(raw_predictions, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != raw.shape or not len(labels):
        raise ValueError("final labels/predictions must be nonempty matching vectors")
    # Preserve nonfinite outputs in the bounded view; clipping infinities must
    # never hide an invalid model behind apparently finite bounded predictions.
    clipped = np.where(np.isfinite(raw), np.clip(raw, 0, 1), raw)
    baseline = np.full(labels.shape, baseline_value, dtype=np.float64)
    views = {"raw": raw, "clipped": clipped, "baseline_train_mean": baseline}
    diagnostics = {name: _diagnostics(values) for name, values in views.items()}
    diagnostics["labels"] = _diagnostics(labels)
    diagnostics["raw_collapse"] = diagnostics["raw"]["constant"]
    diagnostics["clipped_only_constant"] = bool(
        diagnostics["raw"]["all_finite"]
        and not diagnostics["raw_collapse"]
        and diagnostics["clipped"]["constant"]
    )
    metrics = {name: _metrics(labels, values) for name, values in views.items()}
    failures = []
    if not all(diagnostics[name]["all_finite"] for name in (*views, "labels")):
        failures.append("nonfinite labels or predictions")
    if diagnostics["raw_collapse"]:
        failures.append("raw model collapse: constant native predictions")
    if any(value["mae"] is None or value["mse"] is None for value in metrics.values()):
        failures.append("nonfinite error metrics")
    return {
        "validated": not failures,
        "failure_reasons": failures,
        "target_count": len(labels),
        **metrics,
        "diagnostics": diagnostics,
    }


def write_bundle(run, rows, predictions, partition, identity):
    """Preserve even failed numerical evidence; only valid reports can publish."""
    directory = run.local_dir / "final_evaluation"
    directory.mkdir()
    baseline = partition["baseline_train_mean"]
    metrics = prediction_metrics(
        [row["target"] for row in rows], predictions, baseline["value"]
    )
    metrics.update(identity)
    metrics["baseline_fit"] = baseline
    storage.write_verified(directory / "metrics.json", metrics)
    with (directory / "predictions.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_row",
                "source_identity",
                "sequence",
                "sequence_sha256",
                "protospacer",
                "label",
                "raw_prediction",
                "bounded_efficiency",
                "baseline_train_mean",
            ),
        )
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            writer.writerow(
                {
                    "source_row": row["source_row"],
                    "source_identity": canonical_bytes(row["raw"]).decode("utf-8"),
                    "sequence": row["sequence"],
                    "sequence_sha256": row["sequence_sha256"],
                    "protospacer": row["protospacer"],
                    "label": row["target"],
                    "raw_prediction": prediction,
                    "bounded_efficiency": float(np.clip(prediction, 0, 1))
                    if np.isfinite(prediction)
                    else prediction,
                    "baseline_train_mean": baseline["value"],
                }
            )
    report = {
        **identity,
        "validated": metrics["validated"],
        "failure_reasons": metrics["failure_reasons"],
        "test_source_rows": [row["source_row"] for row in rows],
        "prediction_policy": "native raw output and clipping to [0, 1]; fixed before evaluation",
        "comparison_policy": COMPARISON_POLICY,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            name: storage.file_hash(directory / name)
            for name in ("metrics.json", "predictions.csv")
        },
    }
    storage.write_verified(directory / "evaluation.json", report)
    return report
