"""One manifest-driven CNN fit, with its own evidence (#116).

`cas12a/train.py` fits five models on a `KFold` it constructs itself. #115
needs the opposite: one model, on rows somebody else chose, with the Test
partition untouched - because the comparison it draws against NT-Cas12a is
only meaningful if both models saw the same training rows and neither was
selected on the rows the comparison is scored on. Those are different enough
that this is a separate entry point rather than a flag on that one; the
five-fold path, its checkpoint and its metrics JSON are left exactly as they
are, and nothing here calls `train_5fold` or `make_kfold`.

What is shared is the recipe, imported from `cas12a.train` rather than
restated: seed 42, Adam 5e-4 (in `cas12a/model.py`), MSE, batch 32, a ceiling
of 100 epochs, early stopping on `val_loss` with patience 10 and restored best
weights, and `ReduceLROnPlateau` at patience 5 / factor 0.5 / floor 1e-5. A
copy of those numbers here would be a second recipe that looks like the first.

Run `python -m cas12a.cnn_single`. Membership comes from
`cas12a/cnn_manifest.py`, which reads the saved partition and refuses anything
that is not NT's; this module adds the three things a fit has to prove about
itself:

- **one model, one fit.** `SingleFit` raises on a second build or a second
  call, so "we only trained once" is a property of the code rather than a
  claim in a commit message. The count reaches `history.json` either way.
- **the best epoch, separately from the epochs run.** `restore_best_weights`
  means the returned weights are the minimum-`val_loss` epoch's, so that epoch
  is read off the history - a saved checkpoint whose reported epoch no epoch
  produced is the defect `cas12a/docs/cas12a-model-review.md` §3 P2 names.
- **save and reload.** Weight identity is judged exactly, and the forward pass
  at a tolerance derived from FP32's own resolution. Those are two different
  failures - a partially restored graph and reduction-order drift - and #111
  is the issue that was filed because one bound had been made to carry both.

A fixture run (anything but the production 12,000/1,500/1,500 manifest) writes
the same evidence with `records_confirmed_training` false. #117 is the actual
confirmed training run; a rehearsal here is not it, and `status.json` says so.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from cas12a.cnn_manifest import (
    PRODUCTION_IDENTITY,
    fitting_set,
    manifest_counts,
    split_config_for,
)
from cas12a.model import LEARNING_RATE, create_cas12a_model
from cas12a.nt_data import PRODUCTION_COUNTS, digest
from cas12a.nt_storage import write_verified
from cas12a.train import (
    BATCH_SIZE,
    EPOCHS,
    SEED,
    library_versions,
    make_callbacks,
    set_seeds,
    training_configuration,
)

#: Where a run lands, one directory per `run_id`, created fresh. Separate from
#: `cas12a/artifacts/`, which holds the five-fold checkpoint and its metrics:
#: #115 preserves those, and a path that could overwrite them is the easiest
#: way to fail that requirement by accident.
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "artifacts" / "cnn_manifest"

#: The NT run whose partition this path exists to reuse (#110).
DEFAULT_PARTITION = (
    Path(__file__).resolve().parent
    / "models"
    / "NT_cas12a"
    / "result"
    / "20261101-nt500m-full"
    / "partition.json"
)
DEFAULT_SOURCE = (
    Path(__file__).resolve().parent / "dataset" / "kim_2018_cas12a_dataset.xlsx"
)

CHECKPOINT_FILENAME = "cas12a_cnn_manifest.keras"
CONFIGURATION_FILENAME = "configuration.json"
HISTORY_FILENAME = "history.json"
CHECKPOINT_EVIDENCE_FILENAME = "checkpoint.json"
PREDICTIONS_FILENAME = "validation_predictions.csv"
STATUS_FILENAME = "status.json"

#: Half the gap between 1.0 and the next representable FP32 value, and a
#: deliberate margin in multiples of it. Both mirror
#: `cas12a/nt_training.py`'s treatment of the same question and are restated
#: rather than imported, because that module imports `torch` and this one is
#: the TensorFlow side of the package - a cross-import would make a CNN fit
#: depend on the NT runtime being installed. The reason the margin exists is
#: the same there and here: identical weights do not imply an identical kernel
#: or reduction order between a live model and a freshly loaded one. It bounds
#: forward reproducibility only. Loading correctness is asserted exactly, by
#: weight identity, so this number never carries it.
UNIT_ROUNDOFF = 2.0**-24
FORWARD_ULP_BUDGET = 128

#: How many mismatching tensors one failure report names.
MISMATCH_EVIDENCE_LIMIT = 16


class SingleFitViolation(RuntimeError):
    """A second model or a second fit was attempted inside one run.

    Not a `ValueError`: nothing about the inputs is wrong. The run asked to
    train twice, which makes "one seed-42 training" untrue regardless of what
    the second fit produced, so it cannot be recovered from by retrying.
    """


class ReloadMismatch(ValueError):
    """A save/reload comparison failed, carrying its own evidence.

    The payload matters more than the message: by the time anyone reads this,
    the in-memory model is gone and this is the only record able to tell a
    numerical variation apart from a checkpoint that did not load.
    """

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def forward_tolerance():
    """The FP32 bound for comparing two forward passes of the same weights."""
    rtol = FORWARD_ULP_BUDGET * UNIT_ROUNDOFF
    return {
        "precision": "float32",
        "unit_roundoff": UNIT_ROUNDOFF,
        "ulp_budget": FORWARD_ULP_BUDGET,
        "rtol": rtol,
        "atol": rtol / 10.0,
        "basis": (
            "rtol = ulp_budget * FP32 unit roundoff; atol anchors near-zero "
            "predictions at one tenth of rtol. A deliberate margin against "
            "reduction-order drift, not a measured error bound. It bounds "
            "forward reproducibility only: loading correctness rests on exact "
            "weight identity, which is checked separately."
        ),
    }


def _json_number(value):
    """Keep a non-finite magnitude legible: canonical evidence forbids NaN/Inf."""
    value = float(value)
    return value if math.isfinite(value) else repr(value)


def _deviation(actual, expected):
    """Finite, JSON-safe numerical evidence for one prediction comparison."""
    left = np.asarray(actual, dtype=np.float64).ravel()
    right = np.asarray(expected, dtype=np.float64).ravel()
    if left.shape != right.shape:
        return {
            "comparable": False,
            "actual_shape": list(left.shape),
            "expected_shape": list(right.shape),
        }
    difference = np.abs(left - right)
    nonzero = right != 0
    relative = difference[nonzero] / np.abs(right[nonzero])
    return {
        "comparable": True,
        "count": int(left.size),
        "zero_reference_count": int((~nonzero).sum()),
        "max_absolute_difference": _json_number(difference.max()) if left.size else 0.0,
        "max_relative_difference": (
            _json_number(relative.max()) if relative.size else 0.0
        ),
    }


#: What makes two weights the same weight: their position in the model's own
#: weight order, their shape, their dtype and their bytes. `name` is recorded
#: beside them and deliberately not compared - Keras's `Bidirectional` wrapper
#: reports its inner cell's variables as `bidirectional/forward_lstm/...` in a
#: live model and as `forward_lstm/...` in a reloaded one, so a name-keyed
#: comparison would report six renamed tensors as twelve missing and
#: unexpected ones and bury a real defect in the noise.
IDENTITY_KEYS = ("index", "shape", "dtype", "sha256")


def weight_identities(model):
    """Exact per-tensor content digests, in the model's own weight order.

    Raw bytes through a uint8 view, so nothing is cast or rounded on the way
    in and a variable that came back as a fresh initialisation cannot hash the
    same as the one it replaced.
    """
    identities = []
    for index, variable in enumerate(model.weights):
        value = variable.numpy() if hasattr(variable, "numpy") else variable
        array = np.ascontiguousarray(value)
        identities.append(
            {
                "index": index,
                "name": getattr(variable, "path", None) or variable.name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest(),
            }
        )
    return identities


def comparable_identity(entry):
    """The part of a weight identity that a reload has to reproduce."""
    return {key: entry[key] for key in IDENTITY_KEYS}


def _identity_mismatches(reloaded, expected):
    mismatches = []
    for index in range(max(len(expected), len(reloaded))):
        if index >= len(reloaded):
            mismatches.append(
                {"index": index, "reason": "missing_after_reload", **expected[index]}
            )
        elif index >= len(expected):
            mismatches.append(
                {"index": index, "reason": "unexpected_after_reload", **reloaded[index]}
            )
        elif comparable_identity(reloaded[index]) != comparable_identity(
            expected[index]
        ):
            mismatches.append(
                {
                    "index": index,
                    "reason": "content_differs",
                    "expected": expected[index],
                    "reloaded": reloaded[index],
                }
            )
    return mismatches


def verify_weight_identity(model, expected, comparison):
    """Reject a reload whose weights are not bit-identical to the fitted ones."""
    identities = weight_identities(model)
    mismatches = _identity_mismatches(identities, expected)
    if mismatches:
        raise ReloadMismatch(
            "reloaded weights differ from the fitted model",
            {
                "kind": "weight_identity",
                "comparison": comparison,
                "expected_tensors": len(expected),
                "reloaded_tensors": len(identities),
                "mismatch_count": len(mismatches),
                "mismatches": mismatches[:MISMATCH_EVIDENCE_LIMIT],
            },
        )
    return identities


def compare_forward(actual, expected, comparison):
    """Judge forward reproducibility alone, at FP32's own resolution."""
    tolerance = forward_tolerance()
    deviation = _deviation(actual, expected)
    if not deviation["comparable"] or not np.allclose(
        np.ravel(actual),
        np.ravel(expected),
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    ):
        raise ReloadMismatch(
            "reloaded checkpoint predictions differ from the fitted model",
            {
                "kind": "forward_reproducibility",
                "comparison": comparison,
                "tolerance": tolerance,
                "deviation": deviation,
            },
        )
    return {"comparison": comparison, "tolerance": tolerance, "deviation": deviation}


class SingleFit:
    """One model and one `fit`, enforced.

    "Exactly one fresh model, fitted once" is the part of #116 that a later
    reader cannot check from the artifacts alone - two fits leave the same
    files as one. So it is held here instead: the second build or the second
    call raises, and the counts are recorded whatever happens.
    """

    def __init__(self, build=create_cas12a_model):
        self._build = build
        self.model_builds = 0
        self.fit_calls = 0

    def model(self):
        self.model_builds += 1
        if self.model_builds > 1:
            raise SingleFitViolation(
                f"a single-fit run builds one model; build {self.model_builds} "
                "was requested"
            )
        return self._build()

    def fit(self, model, train, validation, *, epochs, verbose):
        self.fit_calls += 1
        if self.fit_calls > 1:
            raise SingleFitViolation(
                f"a single-fit run fits once; call {self.fit_calls} was requested"
            )
        return model.fit(
            [train["X_seq"], train["X_feat"]],
            train["y"],
            # Explicit, never `validation_split`: the rows a model is selected
            # on are the manifest's validation rows, not a tail of the training
            # array that happens to be the right length.
            validation_data=(
                [validation["X_seq"], validation["X_feat"]],
                validation["y"],
            ),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=make_callbacks(),
            verbose=verbose,
        )

    @property
    def counts(self):
        return {"model_builds": self.model_builds, "fit_calls": self.fit_calls}


def predict(model, partition):
    """Validation predictions as a flat float64 array, at the fit batch size.

    The batch size is pinned to the fit's because a different one changes the
    reduction order, which would show up in the reload comparison as drift
    that has nothing to do with loading.
    """
    predictions = model.predict(
        [partition["X_seq"], partition["X_feat"]], batch_size=BATCH_SIZE, verbose=0
    )
    return np.asarray(predictions, dtype=np.float64).ravel()


def fit_single(prepared, *, epochs=EPOCHS, seed=SEED, verbose=2, budget=None):
    """Fit one model on the manifest's rows; return `(model, record)`.

    `record` separates `best_epoch` from `epochs_ran` and carries the whole
    per-epoch history, the validation predictions of the restored best weights
    and the one-model/one-fit counts.
    """
    budget = SingleFit() if budget is None else budget
    train = prepared["partitions"]["train"]
    validation = prepared["partitions"]["validation"]

    set_seeds(seed)
    model = budget.model()
    started = time.perf_counter()
    history = budget.fit(
        model, train, validation, epochs=epochs, verbose=verbose
    ).history
    elapsed = time.perf_counter() - started

    val_loss = [float(value) for value in history["val_loss"]]
    # `restore_best_weights=True` puts back the minimum-`val_loss` epoch, so
    # the reported epoch is `argmin` of the history: it has to describe the
    # weights that are in the model this returns, not where the loop stopped.
    best_epoch = int(np.argmin(val_loss)) + 1
    record = {
        "epochs_max": epochs,
        "epochs_ran": len(val_loss),
        "best_epoch": best_epoch,
        "best_val_loss": val_loss[best_epoch - 1],
        "final_val_loss": val_loss[-1],
        "early_stopped": len(val_loss) < epochs,
        "wall_clock_s": round(elapsed, 1),
        "train_rows": len(train["source_rows"]),
        "validation_rows": len(validation["source_rows"]),
        "history": {
            key: [_json_number(value) for value in values]
            for key, values in history.items()
        },
        "validation_predictions": [
            _json_number(value) for value in predict(model, validation)
        ],
        **budget.counts,
    }
    return model, record


def save_checkpoint(model, run_dir, record, validation):
    """Save, reload, and prove the reload is the same model.

    Order matters. Weight identity is compared first, before any forward pass,
    because a tensor that came back as a fresh initialisation is a loading
    defect however the predictions happen to land - and then the forward pass
    is compared at the FP32 tolerance, which is the only thing that bound is
    allowed to cover (#111).
    """
    path = run_dir / CHECKPOINT_FILENAME
    model.save(path)
    with path.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()

    fitted = weight_identities(model)
    reloaded_model = tf.keras.models.load_model(path)
    identities = verify_weight_identity(
        reloaded_model, fitted, "reload_vs_fitted_weights"
    )
    reloaded = predict(reloaded_model, validation)
    forward = compare_forward(
        reloaded,
        np.asarray(record["validation_predictions"], dtype=np.float64),
        "reload_vs_fitted_validation_predictions",
    )
    return reloaded, {
        "filename": CHECKPOINT_FILENAME,
        "sha256": checkpoint_sha256,
        "bytes": path.stat().st_size,
        "best_epoch": record["best_epoch"],
        "selected_by": "lowest validation loss, restored by EarlyStopping",
        "weight_tensors": len(identities),
        "weight_identities": identities,
        "weight_identity_verified": True,
        "forward_reproducibility": forward,
    }


def cpu_settings():
    """What this process was allowed to use, as the record reports it."""
    return {
        "intra_op_parallelism_threads": (
            tf.config.threading.get_intra_op_parallelism_threads()
        ),
        "inter_op_parallelism_threads": (
            tf.config.threading.get_inter_op_parallelism_threads()
        ),
        "visible_devices": sorted(
            device.device_type for device in tf.config.list_physical_devices()
        ),
        "processor": platform.processor(),
    }


def configuration(manifest, prepared, source, partition_path, *, epochs, seed):
    """The frozen settings and input identity this run is answerable to."""
    counts = manifest_counts(manifest)
    run_kind = split_config_for(manifest).run_kind
    recipe = training_configuration()
    return {
        "issue": 116,
        "study": 115,
        "serialization": "cas12a-cnn-single-v1",
        "entry_point": "cas12a.cnn_single",
        "run_kind": run_kind,
        "source": {
            "path": str(source),
            "file_sha256": manifest["source_file_sha256"],
            "table_sha256": manifest["table_sha256"],
            "usable_rows": len(manifest["rows"]),
        },
        "partition": {
            "path": str(partition_path),
            "manifest_sha256": digest(manifest),
            "partition_sha256": manifest["partition_sha256"],
            "training_partition_sha256": manifest["training_partition_sha256"],
            "algorithm": manifest["algorithm"],
            "seed": manifest["seed"],
            "counts": dict(zip(("train", "validation", "test"), counts)),
            "reused_without_resplitting": True,
            "pinned_to_nt_production_identity": run_kind == "production",
            "expected_production_identity": (
                PRODUCTION_IDENTITY if run_kind == "production" else None
            ),
        },
        "membership": {
            "train_source_rows": prepared["partitions"]["train"]["source_rows"],
            "validation_source_rows": (
                prepared["partitions"]["validation"]["source_rows"]
            ),
            "held_out": prepared["held_out"],
            "held_out_used_for_fitting_or_selection": False,
        },
        "training": {
            **recipe,
            "epochs_max": epochs,
            "seed": seed,
            "learning_rate": LEARNING_RATE,
            "loss": "mse",
            "optimizer": "adam",
            "folds": 1,
            "validation_source": "manifest validation membership, passed explicitly",
        },
        "library_versions": library_versions(),
        "cpu_settings": cpu_settings(),
    }


def write_predictions(path, prepared, record, reloaded):
    """Per-row validation predictions, joined by `source_row`, not by offset.

    `sequence_sha256` travels with every row for the same reason: #115 reads
    this file next to NT's own predictions, and an array offset is not an
    identity two runs can be joined on.
    """
    validation = prepared["partitions"]["validation"]
    fitted = np.asarray(record["validation_predictions"], dtype=np.float64)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_row",
                "sequence_sha256",
                "target",
                "prediction",
                "reload_prediction",
                "absolute_difference",
            ]
        )
        for offset, identity in enumerate(validation["source_rows"]):
            row = prepared["by_identity"][identity]
            writer.writerow(
                [
                    identity,
                    row["sequence_sha256"],
                    repr(float(row["target"])),
                    repr(float(fitted[offset])),
                    repr(float(reloaded[offset])),
                    repr(abs(float(reloaded[offset]) - float(fitted[offset]))),
                ]
            )
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def status(config, record, checkpoint, predictions_sha256):
    """What this run is, and what it is not.

    `records_confirmed_training` is the field that keeps a rehearsal from being
    read as #117's result. A fixture manifest, a lowered epoch ceiling or
    another seed all make it false, and the `incomplete` list says which.
    """
    reasons = []
    if config["run_kind"] != "production":
        reasons.append(
            f"{config['run_kind']} manifest counts "
            f"{tuple(config['partition']['counts'].values())}, not "
            f"{PRODUCTION_COUNTS}"
        )
    if record["epochs_max"] != EPOCHS:
        reasons.append(
            f"epoch ceiling {record['epochs_max']} overrode the recipe's {EPOCHS}"
        )
    if config["training"]["seed"] != SEED:
        reasons.append(f"seed {config['training']['seed']} is not the recipe's {SEED}")
    return {
        "serialization": config["serialization"],
        "issue": config["issue"],
        "run_kind": config["run_kind"],
        "stages": {
            "manifest_verified": True,
            "membership_identity_verified": True,
            "held_out_excluded": True,
            "single_fit": record["fit_calls"] == 1 and record["model_builds"] == 1,
            "best_weights_restored": 1 <= record["best_epoch"] <= record["epochs_ran"],
            "checkpoint_saved": True,
            "reload_weight_identity_verified": checkpoint["weight_identity_verified"],
            "reload_forward_verified": True,
        },
        "records_confirmed_training": not reasons,
        "not_confirmed_because": reasons,
        "note": (
            "Confirmed single-recipe training for study #115."
            if not reasons
            else "Implementation evidence only; #117 performs the confirmed "
            "training run. Do not report this as that run's result."
        ),
        "artifacts": {
            CONFIGURATION_FILENAME: digest(config),
            HISTORY_FILENAME: digest(record),
            CHECKPOINT_EVIDENCE_FILENAME: digest(checkpoint),
            PREDICTIONS_FILENAME: predictions_sha256,
            CHECKPOINT_FILENAME: checkpoint["sha256"],
        },
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run(
    source,
    partition_path,
    run_dir,
    *,
    epochs=EPOCHS,
    seed=SEED,
    verbose=2,
):
    """Verify, fit once, save, reload, and write the run's evidence.

    `run_dir` must not exist: a run that could land on top of another one
    cannot say which fit its checkpoint came from, and #115 requires a fresh
    result directory per run.
    """
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    manifest = json.loads(Path(partition_path).read_text(encoding="utf-8"))
    prepared = fitting_set(source, manifest)
    config = configuration(
        manifest, prepared, source, partition_path, epochs=epochs, seed=seed
    )

    model, record = fit_single(prepared, epochs=epochs, seed=seed, verbose=verbose)
    run_dir.mkdir(parents=True)
    reloaded, checkpoint = save_checkpoint(
        model, run_dir, record, prepared["partitions"]["validation"]
    )
    predictions_sha256 = write_predictions(
        run_dir / PREDICTIONS_FILENAME, prepared, record, reloaded
    )

    report = status(config, record, checkpoint, predictions_sha256)
    for name, value in (
        (CONFIGURATION_FILENAME, config),
        (HISTORY_FILENAME, record),
        (CHECKPOINT_EVIDENCE_FILENAME, checkpoint),
        (STATUS_FILENAME, report),
    ):
        write_verified(run_dir / name, value)
    return {
        "run_dir": run_dir,
        "configuration": config,
        "history": record,
        "checkpoint": checkpoint,
        "status": report,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Kim 2018 table the manifest was written from (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--partition",
        type=Path,
        default=DEFAULT_PARTITION,
        help=f"Saved NT partition manifest to reuse (default: {DEFAULT_PARTITION}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Parent of the fresh run directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run directory name; defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            f"Epoch ceiling; early stopping usually ends sooner (default: "
            f"{EPOCHS}). Lowering it makes the run unreportable, and "
            f"{STATUS_FILENAME} records that."
        ),
    )
    parser.add_argument(
        "--verbose", type=int, default=2, help="Keras fit verbosity (default: 2)."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for path in (args.source, args.partition):
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%d-%H%M%S-cnn-manifest"
    )

    result = run(
        args.source,
        args.partition,
        args.output_root / run_id,
        epochs=args.epochs,
        seed=SEED,
        verbose=args.verbose,
    )
    history, report = result["history"], result["status"]
    print(
        "\n".join(
            [
                f"\nRun directory: {result['run_dir']}",
                (
                    f"Fitted {history['train_rows']} training rows, selected "
                    f"on {history['validation_rows']} validation rows "
                    f"({report['run_kind']} manifest)."
                ),
                (
                    f"Best epoch {history['best_epoch']} of "
                    f"{history['epochs_ran']} run, val_loss "
                    f"{history['best_val_loss']:.6f} "
                    f"({history['wall_clock_s']:.1f}s)."
                ),
                (
                    f"Checkpoint {result['checkpoint']['filename']} sha256 "
                    f"{result['checkpoint']['sha256'][:16]}..., reload verified."
                ),
            ]
        ),
        flush=True,
    )
    if not report["records_confirmed_training"]:
        print(
            "NOTE: this run is not the confirmed #115 training. "
            + "; ".join(report["not_confirmed_because"]),
            flush=True,
        )
    return result


if __name__ == "__main__":
    main()
