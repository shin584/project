"""Research protocol for fresh 9-Hydra validation runs (issue #51).

Planning and report-only inspection never import a training backend. Selection
freeze and final Test evaluation are explicit, closed stages until later tickets.
"""

import json
import platform
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from SpCas9.spcas9_data import HEAD_NAMES, library_versions, output_name
from SpCas9.spcas9_eval import build_predictions, build_report
from SpCas9.spcas9_execution import reserve_destination, source_record
from SpCas9.spcas9_metrics import predictions_by_head
from SpCas9.spcas9_selection import (
    NGG_MAE,
    ngg_stopping,
    ngg_summary,
    validation_ngg_mask,
)
from SpCas9.spcas9_split import file_sha256, write_json
from SpCas9.spcas9_study_inputs import REQUIRED_ROWS, InputPins, StudyInputs
from SpCas9.spcas9_train import (
    CHECKPOINT_FILENAME,
    REFERENCE_LOSS,
    encode_inputs,
    hyperparameter_record,
    masked_labels_and_weights,
    reference_validation_loss,
)

if TYPE_CHECKING:
    from SpCas9.spcas9_study_backend import TrainingBackend

__all__ = ["InputPins", "StudyInputs", "StudyProtocol"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("study/run identifier must be a single safe path component")
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _state(path: Path, value: dict) -> None:
    temporary = path.with_name(f".state-{uuid.uuid4().hex}.tmp")
    write_json(value, temporary)
    temporary.replace(path)


def _validation_report(frame: pd.DataFrame, *, metrics_schema_version: int = 2) -> dict:
    report = build_report(frame, metrics_schema_version=metrics_schema_version)
    del report["testset"]
    return {**report, "split": "val", "population": "9-Hydra validation"}


def _predictions(outputs: dict, rows: int) -> dict[str, np.ndarray]:
    predictions = predictions_by_head(outputs)
    for head, values in predictions.items():
        if len(values) != rows or not np.isfinite(values).all():
            raise ValueError(
                f"{head}: non-finite or incorrectly shaped required predictions"
            )
    return predictions


def _stopping(
    history: dict[str, list[float]], *, selection: str = "reference_mse"
) -> dict:
    if selection == "ngg":
        stopping = ngg_stopping(history)
        for name in (
            "recipe_optimization_loss",
            "val_recipe_optimization_loss",
            "learning_rate",
        ):
            values = np.asarray(history.get(name, []), dtype=float)
            if (
                values.ndim != 1
                or len(values) != stopping["epochs_run"]
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"invalid {name} history")
        return {**stopping, "early_stopped": stopping["epochs_run"] < 50}
    reference = np.asarray(history[REFERENCE_LOSS], dtype=float)
    if not 1 <= len(reference) <= 50 or not np.isfinite(reference).all():
        raise ValueError("invalid reference validation stopping history")
    for key in (
        "recipe_optimization_loss",
        "val_recipe_optimization_loss",
        "learning_rate",
    ):
        values = np.asarray(history[key], dtype=float)
        if len(values) != len(reference) or not np.isfinite(values).all():
            raise ValueError(f"invalid {key} history")
    return {
        "monitor": REFERENCE_LOSS,
        "epochs_run": len(reference),
        "best_epoch": int(np.argmin(reference)) + 1,
        "best_reference_loss": float(np.min(reference)),
        "early_stopped": len(reference) < 50,
        "restore_best_weights": True,
    }


def _protocol(inputs: StudyInputs, checkpoint_selection: str = "reference_mse") -> dict:
    if checkpoint_selection not in ("ngg", "reference_mse"):
        raise ValueError("unknown checkpoint selection policy")
    params = hyperparameter_record()
    for key in ("early_stopping", "reduce_lr_on_plateau"):
        params[key]["monitor"] = (
            NGG_MAE if checkpoint_selection == "ngg" else REFERENCE_LOSS
        )
    if checkpoint_selection == "ngg":
        params["early_stopping"]["restore_best_weights"] = False
    record = {
        "schema_version": 1,
        "issue": 51,
        "model": "9-Hydra v2",
        "architecture": {
            "trunk": [
                "Conv1D(128,3,relu,same)",
                "BatchNormalization",
                "LSTM(64)",
                "Dropout(0.3)",
                "Dense(64,relu)",
            ],
            "each_head": ["Dense(32,relu)", "Dense(1,linear)"],
        },
        "rows": dict(REQUIRED_ROWS),
        "inputs": inputs.record(),
        "head_names": list(HEAD_NAMES),
        "predictor_input": "normalized 30bp sequence only, one-hot A/C/G/T",
        "label_scale": "published percentage / 100; no clipping",
        "head_coefficients": {head: 1.0 for head in HEAD_NAMES},
        "training_weights": "binary measured-label mask; metadata does not condition sampling or weights",
        "hyperparameters": params,
        "checkpoint_selection": checkpoint_selection,
        "reference_validation_loss": {
            "name": REFERENCE_LOSS,
            "definition": "sum of all nine binary-masked squared-error sums / validation rows",
        },
        "optimization_loss": "recipe_optimization_loss",
        "save_reload_parity": {"split": "val", "rtol": 1e-5, "atol": 1e-6},
        "training_seeds": [42, 43, 44],
        "implemented_stages": ["plan", "train_validation", "report_only"],
        "deferred_stages": ["freeze_selection", "evaluate_test"],
    }
    if checkpoint_selection == "ngg":
        record.update(schema_version=2, issue=84)
        del record["reference_validation_loss"]
        record["checkpoint_policy"] = {
            "monitor": NGG_MAE,
            "definition": "equal-head measured NGG MAE; exact ties prefer higher mean NGG Spearman, then earlier epoch",
            "weights_owner": "NGG selector",
            "required_monitor_failure": "missing, stale or non-finite values fail the run",
        }
    return record


class StudyProtocol:
    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def plan(
        cls,
        output_root: Path,
        inputs: StudyInputs | None = None,
        *,
        study_id: str | None = None,
        checkpoint_selection: str = "reference_mse",
    ) -> "StudyProtocol":
        inputs = inputs or StudyInputs()
        inputs.validate()
        protocol = _protocol(inputs, checkpoint_selection)
        path = Path(output_root) / _identifier(study_id or f"study-{uuid.uuid4().hex}")
        reserve_destination(path)
        protocol_path = write_json(protocol, path / "protocol.json")
        write_json(
            {
                "created_at": _now(),
                "protocol_sha256": file_sha256(protocol_path),
                "source": source_record(),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "libraries": library_versions(),
                },
            },
            path / "study.json",
        )
        return cls(path)

    def _protocol(self) -> dict:
        study = _read(self.path / "study.json")
        if file_sha256(self.path / "protocol.json") != study["protocol_sha256"]:
            raise ValueError("corrupted protocol evidence")
        return _read(self.path / "protocol.json")

    def plan_run(self, *, seed: int, run_id: str | None = None) -> str:
        protocol = self._protocol()
        if seed not in protocol["training_seeds"]:
            raise ValueError("training seed must be 42, 43 or 44")
        run_id = _identifier(run_id or f"v2-{seed}-{uuid.uuid4().hex}")
        path = self.path / "runs" / run_id
        reserve_destination(path)
        write_json(
            {"recipe": "v2", "seed": seed, "created_at": _now()}, path / "plan.json"
        )
        write_json(
            {
                "status": "planned",
                "eligible": False,
                "plan_sha256": file_sha256(path / "plan.json"),
                "study_sha256": file_sha256(self.path / "study.json"),
            },
            path / "state.json",
        )
        return run_id

    def train_validation(
        self, run_id: str, *, backend: "TrainingBackend | None" = None
    ) -> dict:
        """One reserved fit; interrupted/failed evidence remains explicitly ineligible."""
        from SpCas9.spcas9_study_backend import KerasV2Backend, TrainingRows

        self.report_only(run_id)
        path = self.path / "runs" / _identifier(run_id)
        state = _read(path / "state.json")
        if state["status"] != "planned" or {p.name for p in path.iterdir()} != {
            "plan.json",
            "state.json",
        }:
            raise FileExistsError(f"run target already used: {path}")
        # Exclusive reservation also prevents two processes from fitting this run.
        with (path / "started.json").open("x", encoding="utf-8") as f:
            json.dump({"started_at": _now()}, f)
        started = time.monotonic()
        state.update(status="running", eligible=False, started_at=_now())
        _state(path / "state.json", state)
        try:
            protocol = self._protocol()
            frames = StudyInputs.from_record(protocol["inputs"]).validate()
            source = source_record()
            execution = {
                "source": source,
                "inputs": protocol["inputs"],
                "versions": library_versions(),
                "seed": _read(path / "plan.json")["seed"],
                "python": sys.version,
                "platform": platform.platform(),
            }
            write_json(execution, path / "execution.json")

            def training_rows(name: str) -> TrainingRows:
                labels, weights = masked_labels_and_weights(frames[name])
                return TrainingRows(encode_inputs(frames[name]), labels, weights)

            training, validation = training_rows("train"), training_rows("val")
            checkpoint = path / CHECKPOINT_FILENAME
            selection = protocol.get("checkpoint_selection", "reference_mse")
            backend = backend or KerasV2Backend(selection=selection)
            if isinstance(backend, KerasV2Backend) and backend.selection != selection:
                raise ValueError(
                    "backend checkpoint selection differs from frozen protocol"
                )
            result = backend.fit(
                training,
                validation,
                seed=_read(path / "plan.json")["seed"],
                checkpoint_path=checkpoint,
            )
            checkpoint_digest = file_sha256(checkpoint)
            stopping = _stopping(result.history, selection=selection)
            before = _predictions(result.validation_predictions, len(frames["val"]))
            after = _predictions(
                backend.predict(checkpoint, validation.inputs), len(frames["val"])
            )
            for head in HEAD_NAMES:
                if not np.allclose(before[head], after[head], rtol=1e-5, atol=1e-6):
                    raise ValueError(f"validation save/reload parity failed for {head}")
            if checkpoint_digest != file_sha256(checkpoint):
                raise ValueError("checkpoint evidence changed during parity")
            outputs = {output_name(head): values for head, values in after.items()}
            if selection == "ngg":
                restored = ngg_summary(
                    validation.labels,
                    validation.weights,
                    outputs,
                    validation_ngg_mask(validation.inputs),
                )
                for metric in ("mae", "spearman"):
                    if not np.isclose(
                        restored[metric],
                        stopping[f"best_ngg_{metric}"],
                        rtol=1e-5,
                        atol=1e-6,
                    ):
                        raise ValueError(
                            f"checkpoint predictions do not match selected NGG {metric}"
                        )
                expected_selection = ngg_stopping(result.history)
                if result.selection is None or any(
                    result.selection.get(key) != value
                    for key, value in expected_selection.items()
                ):
                    raise ValueError(
                        "backend NGG selection evidence differs from monitor history"
                    )
            else:
                restored_reference = reference_validation_loss(
                    validation.labels, validation.weights, outputs
                )
                if not np.isclose(
                    restored_reference,
                    stopping["best_reference_loss"],
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "checkpoint predictions do not match best reference validation loss"
                    )
            predictions = build_predictions(frames["val"], after)
            predictions.to_csv(
                path / "predictions.csv", index=False, lineterminator="\n"
            )
            # Score exactly the persisted representation used by report-only reconstruction.
            persisted = pd.read_csv(path / "predictions.csv")
            write_json(_validation_report(persisted), path / "validation.json")
            write_json(
                {
                    **execution,
                    "history": result.history,
                    "stopping": stopping,
                    "selection": result.selection,
                    "environment": result.environment,
                    "rows_fitted": {
                        "train": len(frames["train"]),
                        "val": len(frames["val"]),
                    },
                    "parity": {
                        "split": "val",
                        "rows": len(persisted),
                        "rtol": 1e-5,
                        "atol": 1e-6,
                        "passed": True,
                    },
                    "duration_seconds": time.monotonic() - started,
                },
                path / "training.json",
            )
            files = (
                "plan.json",
                "started.json",
                "execution.json",
                CHECKPOINT_FILENAME,
                "predictions.csv",
                "validation.json",
                "training.json",
            )
            write_json(
                {"files": {name: file_sha256(path / name) for name in files}},
                path / "evidence.json",
            )
            state.update(
                status="completed",
                eligible=True,
                evidence_sha256=file_sha256(path / "evidence.json"),
                finished_at=_now(),
            )
            _state(path / "state.json", state)
            return self.report_only(run_id)
        except BaseException as error:
            state.update(
                status="failed",
                eligible=False,
                error=f"{type(error).__name__}: {error}",
                duration_seconds=time.monotonic() - started,
                finished_at=_now(),
            )
            _state(path / "state.json", state)
            raise

    def report_only(self, run_id: str) -> dict:
        protocol = self._protocol()
        path = self.path / "runs" / _identifier(run_id)
        state = _read(path / "state.json")
        if (
            file_sha256(path / "plan.json") != state["plan_sha256"]
            or file_sha256(self.path / "study.json") != state["study_sha256"]
        ):
            raise ValueError("corrupted study/plan evidence")
        report = {**state, "plan": _read(path / "plan.json"), "protocol": protocol}
        if state["status"] != "completed":
            return {**report, "eligible": False}
        if file_sha256(path / "evidence.json") != state["evidence_sha256"]:
            raise ValueError("corrupted evidence manifest")
        evidence = _read(path / "evidence.json")
        for name, expected in evidence["files"].items():
            if name != Path(name).name or file_sha256(path / name) != expected:
                raise ValueError(f"corrupted run evidence: {name}")
        training = _read(path / "training.json")
        persisted_validation = _read(path / "validation.json")
        validation = _validation_report(
            pd.read_csv(path / "predictions.csv"),
            metrics_schema_version=persisted_validation.get(
                "metrics_schema_version", 1
            ),
        )
        if validation != persisted_validation:
            raise ValueError("corrupted validation report")
        return {
            **report,
            "training": training,
            "validation": validation,
            "source": training["source"],
            "inputs": training["inputs"],
            "evidence": evidence,
        }

    def freeze_selection(self) -> None:
        raise NotImplementedError("finalist selection/freeze is a later protocol stage")

    def evaluate_test(self) -> None:
        raise PermissionError(
            "Test prediction requires a valid finalist freeze; stage not implemented"
        )
