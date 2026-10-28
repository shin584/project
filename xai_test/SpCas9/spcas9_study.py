"""Research protocol for fresh 9-Hydra validation runs (issue #51).

Planning and report-only inspection never import a training backend. Final Test
inference requires both passing validation comparisons and an intact finalist freeze.
"""

import json
import os
import platform
import re
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from SpCas9.spcas9_acceptance import GATE_SETTINGS, bootstrap_intervals
from SpCas9.spcas9_acceptance import build_report as build_validation_report
from SpCas9.spcas9_budget import (
    BudgetedBackend,
    BudgetLedger,
    configuration,
    environment_record,
    read,
    synthetic_handoff,
)
from SpCas9.spcas9_data import HEAD_NAMES, library_versions, output_name
from SpCas9.spcas9_eval import (
    PREDICTIONS_CHECKSUM_FILENAME,
    _read_predictions,
    build_predictions,
    build_report,
    validate_prediction_rows,
)
from SpCas9.spcas9_execution import reserve_destination, source_record
from SpCas9.spcas9_metrics import population_metrics, predictions_by_head
from SpCas9.spcas9_selection import (
    NGG_MAE,
    ngg_stopping,
    ngg_summary,
    validation_ngg_mask,
)
from SpCas9.spcas9_split import file_sha256, write_json
from SpCas9.spcas9_study_inputs import (
    REQUIRED_ROWS,
    InputPins,
    StudyInputs,
    checked_bytes,
)
from SpCas9.spcas9_train import (
    CHECKPOINT_FILENAME,
    REFERENCE_LOSS,
    encode_inputs,
    hyperparameter_record,
    masked_labels_and_weights,
    reference_validation_loss,
    split_record,
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


def _validation_run_evidence(run_id: str, run: dict) -> dict:
    """One representation for producing and verifying required validation evidence."""
    return {
        "run_id": run_id,
        "prediction_sha256": run["evidence"]["files"]["predictions.csv"],
        "checkpoint_sha256": run["evidence"]["files"][CHECKPOINT_FILENAME],
        "manifest": run["evidence"],
        "history": run["training"]["history"],
        "stopping": run["training"]["stopping"],
        "environment": run["training"]["environment"],
        "training_ngg": run.get("training_ngg"),
    }


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
    record: dict = {
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
        record.update(schema_version=2, issue=87)
        record["implemented_stages"] += [
            "freeze_selection",
            "evaluate_test",
            "test_report",
        ]
        record["deferred_stages"] = []
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
        execution_mode: str | None = None,
        synthetic_ledger: Path | None = None,
        verification_stored_checkpoint: Path | None = None,
        retry_from: Path | None = None,
        retry_authorization: str | None = None,
    ) -> "StudyProtocol":
        if (retry_from is None) != (retry_authorization is None):
            raise ValueError(
                "retry parent and explicit authorization are both required"
            )
        if retry_from is not None and execution_mode is None:
            raise ValueError("retry requires a controlled execution mode")
        if (
            verification_stored_checkpoint is not None
            and execution_mode != "verification"
        ):
            raise ValueError(
                "disposable stored-checkpoint identity requires verification mode"
            )
        inputs = inputs or StudyInputs()
        inputs.validate()
        protocol = _protocol(inputs, checkpoint_selection)
        with ExitStack() as ownership:
            config = None
            retry = None
            if execution_mode is not None:
                if synthetic_ledger is None:
                    raise ValueError("synthetic ledger evidence is required")
                if retry_from is not None:
                    from SpCas9.spcas9_retry import retry_handoff

                    synthetic, retry = ownership.enter_context(
                        retry_handoff(
                            Path(retry_from),
                            Path(synthetic_ledger),
                            retry_authorization or "",
                        )
                    )
                    if execution_mode != retry["parent_mode"]:
                        raise ValueError(
                            "retry must retain the original execution mode"
                        )
                else:
                    synthetic = ownership.enter_context(
                        synthetic_handoff(Path(synthetic_ledger))
                    )
                config = configuration(protocol, execution_mode, synthetic)
                if retry is not None:
                    config["retry"] = retry
                if verification_stored_checkpoint is not None:
                    config["acceptance"]["historical_checkpoint_sha256"] = file_sha256(
                        Path(verification_stored_checkpoint)
                    )
            path = Path(output_root) / _identifier(
                study_id or f"study-{uuid.uuid4().hex}"
            )
            reserve_destination(
                path,
                (inputs.source_path, inputs.split_dir)
                + ((Path(retry["parent_study"]),) if retry else ()),
            )
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
            if config is not None:
                config_path = write_json(config, path / "execution-config.json")
                study = read(path / "study.json")
                study["execution_config_sha256"] = file_sha256(config_path)
                _state(path / "study.json", study)
                BudgetLedger(path).initialize(
                    file_sha256(config_path),
                    retry["carried_training_seconds"]
                    if retry
                    else config["synthetic"]["training_seconds"],
                )
                owner_path = (
                    Path(retry["parent_study"]) / "authorized-retry.json"
                    if retry
                    else Path(config["synthetic"]["path"] + ".study-owner.json")
                )
                with owner_path.open("x", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "study": str(path.resolve()),
                            "config_sha256": file_sha256(config_path),
                            **(
                                {"authorization": retry["authorization"]}
                                if retry
                                else {}
                            ),
                        },
                        handle,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
        return cls(path)

    def execution_configuration(self) -> dict:
        """Inspect the immutable draft without importing a training backend."""
        BudgetLedger(self.path).events()
        config = read(self.path / "execution-config.json")
        if config["protocol"] != self._protocol():
            raise ValueError("execution/protocol identity mismatch")
        if "retry" in config:
            from SpCas9.spcas9_retry import verify_retry

            verify_retry(config, self.path)
        return config

    def freeze_execution(
        self, *, readiness: Path | None = None, stored_checkpoint: Path | None = None
    ) -> dict:
        """Seal verified essential checks and input/environment identities before fitting."""
        from SpCas9.spcas9_readiness import freeze_execution

        return freeze_execution(self, readiness, stored_checkpoint)

    def budget_status(self) -> dict:
        ledger = BudgetLedger(self.path)
        with ledger.locked():
            self.execution_configuration()
            return ledger.status()

    @contextmanager
    def evaluation_timing(self, *, bootstrap: bool = False):
        """Keep report/evaluation/bootstrap work outside the training allowance."""
        ledger = BudgetLedger(self.path)
        with ledger.locked():
            self.execution_configuration()
            started = time.monotonic()
            try:
                yield
            finally:
                ledger.append(
                    {
                        "kind": "bootstrap" if bootstrap else "evaluation",
                        "seconds": time.monotonic() - started,
                    }
                )

    def recover_reports(self, destination: Path) -> dict:
        """Reconcile an abandoned attempt and copy verified reports to a fresh location.

        A missing child termination receipt means unknown training time. Even a
        known upper bound never authorizes another fit after an interruption.
        Earlier evidence, including incomplete runs, is preserved verbatim.
        """
        ledger = BudgetLedger(self.path)
        with ledger.locked():
            config = self.execution_configuration()
            status = ledger.status()
            if status["attempts"] and status["attempts"][-1]["status"] == "running":
                attempt = status["attempts"][-1]
                receipt = self.path / "runs" / attempt["run_id"] / "fit-ended.json"
                elapsed = None
                if receipt.exists():
                    evidence = read(receipt)
                    value = evidence["elapsed_seconds"]
                    if (
                        not isinstance(value, (int, float))
                        or not np.isfinite(value)
                        or value < 0
                    ):
                        raise ValueError("invalid termination elapsed bound")
                    elapsed = max(attempt["elapsed_seconds"], value)
                ledger.append(
                    {
                        "kind": "recovered",
                        "status": "interrupted",
                        "elapsed_seconds": elapsed,
                        "accounting": "verified child termination upper bound"
                        if elapsed is not None
                        else "unknown elapsed bound; training blocked",
                    }
                )
                status = ledger.status()
            completed = {}
            for attempt in status["attempts"]:
                report = self._report_only(attempt["run_id"])
                if report["eligible"]:
                    completed[attempt["run_id"]] = report
            inputs = StudyInputs.from_record(config["protocol"]["inputs"])
            reserve_destination(
                Path(destination), (self.path, inputs.source_path, inputs.split_dir)
            )
            recovered = {
                "source_study": str(self.path.resolve()),
                "execution_config_sha256": file_sha256(
                    self.path / "execution-config.json"
                ),
                "budget": status,
                "completed": completed,
                "training_permitted": False,
            }
            write_json(recovered, Path(destination) / "recovery.json")
            return recovered

    def _protocol(self) -> dict:
        study = _read(self.path / "study.json")
        if file_sha256(self.path / "protocol.json") != study["protocol_sha256"]:
            raise ValueError("corrupted protocol evidence")
        if (
            "execution_config_sha256" in study
            and file_sha256(self.path / "execution-config.json")
            != study["execution_config_sha256"]
        ):
            raise ValueError("changed execution configuration")
        return _read(self.path / "protocol.json")

    def _controlled(self) -> bool:
        return "execution_config_sha256" in read(self.path / "study.json")

    def plan_run(
        self, *, seed: int, run_id: str | None = None, recipe: str = "v2"
    ) -> str:
        protocol = self._protocol()
        controlled = self._controlled()
        if controlled:
            config = self.execution_configuration()
            if {"recipe": recipe, "seed": seed} not in config["run_order"]:
                raise ValueError("fixed recipe/seed required")
        elif recipe != "v2":
            raise ValueError(
                "ranking recipes require a controlled execution configuration"
            )
        if seed not in protocol["training_seeds"]:
            raise ValueError("training seed must be 42, 43 or 44")
        run_id = _identifier(run_id or f"v2-{seed}-{uuid.uuid4().hex}")
        path = self.path / "runs" / run_id
        reserve_destination(path)
        write_json(
            {"recipe": recipe, "seed": seed, "created_at": _now()}, path / "plan.json"
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
        if self._controlled():
            return self._controlled_train(run_id, backend=backend)
        from SpCas9.spcas9_study_backend import KerasV2Backend

        if backend is None or isinstance(backend, KerasV2Backend):
            raise PermissionError(
                "full fits require fixed execution configuration and verified readiness"
            )
        return self._train_validation(run_id, backend=backend)

    def _controlled_train(self, run_id: str, *, backend) -> dict:
        from SpCas9.spcas9_study_backend import KerasNggBackend, KerasV2Backend

        ledger = BudgetLedger(self.path)
        with ledger.locked():
            config = self.execution_configuration()
            if config["mode"] == "full":
                from SpCas9.spcas9_readiness import verify_execution

                try:
                    verify_execution(self)
                except FileNotFoundError as error:
                    raise PermissionError(
                        "final pre-fit readiness freeze is required"
                    ) from error
            elif backend is None or isinstance(backend, KerasV2Backend):
                raise PermissionError(
                    "verification mode requires a disposable fake backend"
                )
            if (
                source_record() != config["source"]
                or environment_record() != config["environment"]
            ):
                raise ValueError("changed source/environment configuration")
            synthetic = config["synthetic"]
            if file_sha256(Path(synthetic["path"])) != synthetic["sha256"]:
                raise ValueError("changed synthetic evidence")
            if "retry" in config:
                from SpCas9.spcas9_retry import verify_retry

                verify_retry(config, self.path)
            elif read(Path(synthetic["path"] + ".study-owner.json")) != {
                "study": str(self.path.resolve()),
                "config_sha256": file_sha256(self.path / "execution-config.json"),
            }:
                raise ValueError(
                    "synthetic history belongs to another study; budget reset refused"
                )
            self._report_only(run_id)
            status = ledger.status()
            if status["blocked"]:
                raise ValueError(
                    "attempt history blocks all subsequent training; no retry/resume"
                )
            index = len(status["attempts"])
            if index >= 6:
                raise ValueError("six-fit budget exhausted")
            plan = read(self.path / "runs" / _identifier(run_id) / "plan.json")
            identity = {key: plan[key] for key in ("recipe", "seed")}
            if identity != config["run_order"][index]:
                raise ValueError("fixed baseline/candidate seed run order required")
            if config["mode"] == "full":
                required_backend = (
                    KerasV2Backend
                    if identity["recipe"] == "baseline"
                    else KerasNggBackend
                )
                backend = backend or required_backend()
                if (
                    type(backend) is not required_backend
                    or backend.epochs != 50
                    or backend.selection != "ngg"
                    or (
                        isinstance(backend, KerasNggBackend)
                        and backend.auxiliary_weight != 0.01
                    )
                ):
                    raise ValueError("backend differs from the frozen full-fit recipe")
            if status["remaining_seconds"] <= 0:
                raise ValueError("training time budget exhausted")
            for previous in status["attempts"]:
                if not self._report_only(previous["run_id"])["eligible"]:
                    raise ValueError("previous completed run evidence is missing")
            run_path = self.path / "runs" / run_id
            if (
                read(run_path / "state.json")["status"] != "planned"
                or (run_path / "started.json").exists()
            ):
                raise FileExistsError("run already attempted; no retry/resume")
            receipt = {"run_id": run_id, **identity}
            with (self.path / f"attempt-{index + 1}.json").open(
                "x", encoding="utf-8"
            ) as handle:
                json.dump(receipt, handle)
                handle.flush()
                os.fsync(handle.fileno())
            started = time.monotonic()
            ledger.append({"kind": "started", **receipt, "started_at": _now()})
            guarded = BudgetedBackend(
                backend, ledger, started, status["remaining_seconds"]
            )
            try:
                result = self._train_validation(run_id, backend=guarded)
                if guarded.fit_seconds is None:
                    raise ValueError("missing elapsed training evidence")
                # Persist completion only after all run evidence and parity pass.
                ledger.append(
                    {
                        "kind": "finished",
                        "status": "completed",
                        "elapsed_seconds": guarded.fit_seconds,
                    }
                )
                ledger.append(
                    {
                        "kind": "evaluation",
                        "seconds": time.monotonic() - started - guarded.fit_seconds,
                    }
                )
                return result
            except (Exception, KeyboardInterrupt, SystemExit) as error:
                elapsed = time.monotonic() - started
                state = (
                    "timed_out"
                    if isinstance(error, TimeoutError)
                    else "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "failed"
                )
                ledger.append(
                    {
                        "kind": "finished",
                        "status": state,
                        "elapsed_seconds": guarded.fit_seconds
                        if guarded.fit_seconds is not None
                        else elapsed,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if guarded.fit_seconds is not None:
                    ledger.append(
                        {"kind": "evaluation", "seconds": elapsed - guarded.fit_seconds}
                    )
                raise

    def _train_validation(
        self, run_id: str, *, backend: "TrainingBackend | None" = None
    ) -> dict:
        from SpCas9.spcas9_study_backend import KerasV2Backend, TrainingRows

        self._report_only(run_id)
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
            frozen = self.execution_configuration() if self._controlled() else None
            if frozen is not None and (
                source != frozen["source"]
                or environment_record() != frozen["environment"]
            ):
                raise ValueError(
                    "consumed source/environment differs from frozen configuration"
                )
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
            files: tuple[str, ...] = (
                "plan.json",
                "started.json",
                "execution.json",
                CHECKPOINT_FILENAME,
                "predictions.csv",
                "validation.json",
                "training.json",
            )
            if frozen is not None and "reporting" in frozen:
                train_predictions = build_predictions(
                    frames["train"],
                    _predictions(
                        backend.predict(checkpoint, training.inputs),
                        len(frames["train"]),
                    ),
                )
                train_predictions.to_csv(
                    path / "training_predictions.csv", index=False, lineterminator="\n"
                )
                write_json(
                    population_metrics(pd.read_csv(path / "training_predictions.csv"))[
                        "ngg"
                    ],
                    path / "training_ngg.json",
                )
                files += ("training_predictions.csv", "training_ngg.json")
            if file_sha256(checkpoint) != checkpoint_digest:
                raise ValueError(
                    "selected checkpoint changed during training diagnostics"
                )
            write_json(
                {"files": {name: file_sha256(path / name) for name in files}},
                path / "evidence.json",
            )
            if frozen is not None and (
                source_record() != frozen["source"]
                or environment_record() != frozen["environment"]
            ):
                raise ValueError("source/environment changed during training execution")
            state.update(
                status="completed",
                eligible=True,
                evidence_sha256=file_sha256(path / "evidence.json"),
                finished_at=_now(),
            )
            _state(path / "state.json", state)
            return self._report_only(run_id)
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
        if self._controlled():
            with self.evaluation_timing():
                return self._report_only(run_id)
        return self._report_only(run_id)

    def _report_only(self, run_id: str) -> dict:
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
        required = {
            "plan.json",
            "started.json",
            "execution.json",
            CHECKPOINT_FILENAME,
            "predictions.csv",
            "validation.json",
            "training.json",
        }
        if "training_ngg.json" in evidence["files"]:
            required |= {"training_predictions.csv", "training_ngg.json"}
        if set(evidence["files"]) != required:
            raise ValueError("missing or unexpected required run evidence")
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
        training_ngg = None
        if "training_ngg.json" in evidence["files"]:
            training_ngg = population_metrics(
                pd.read_csv(path / "training_predictions.csv")
            )["ngg"]
            if training_ngg != _read(path / "training_ngg.json"):
                raise ValueError("corrupted training NGG report")
        return {
            **report,
            "training": training,
            "validation": validation,
            "source": training["source"],
            "inputs": training["inputs"],
            "evidence": evidence,
            **({"training_ngg": training_ngg} if training_ngg is not None else {}),
        }

    def validation_report(
        self,
        destination: Path,
        *,
        stored_predictions: Path | None = None,
        stored_sidecar: Path | None = None,
    ) -> dict:
        """Rebuild both #86 gates and supplementary CI from verified validation bytes.

        Missing or damaged required run evidence is operationally incomplete.
        No backend is loaded, no candidate Test predictions are accessed and
        no historical seeds are manufactured for the stored-v2 comparison.
        """
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        if stored_sidecar is not None and stored_predictions is None:
            raise ValueError("stored_sidecar requires stored_predictions")
        with self.evaluation_timing():
            config = self.execution_configuration()
            if any(
                config["acceptance"][key] != value
                for key, value in GATE_SETTINGS.items()
            ):
                raise ValueError(
                    "frozen acceptance thresholds differ from reporting implementation"
                )
            inputs = StudyInputs.from_record(config["protocol"]["inputs"])
            raw = checked_bytes(
                inputs.split_dir / "val.csv", inputs.pins.files["val.csv"]
            )
            rows = pd.read_csv(BytesIO(raw))
            rows["sequence"] = rows["sequence"].astype(str).str.strip().str.upper()
            sources: tuple[Path, ...] = (
                self.path,
                inputs.source_path,
                inputs.split_dir,
            )
            if stored_predictions is not None:
                stored_predictions = Path(stored_predictions)
                stored_sidecar = (
                    Path(stored_sidecar)
                    if stored_sidecar is not None
                    else stored_predictions.with_name(PREDICTIONS_CHECKSUM_FILENAME)
                )
                sources += (stored_predictions, stored_sidecar)
            reserve_destination(destination, sources)
            runs: dict = {"baseline": {}, "candidate": {}}
            evidence: dict = {"baseline": {}, "candidate": {}}
            unavailable = {}
            for attempt in BudgetLedger(self.path).status()["attempts"]:
                recipe, seed, run_id = (
                    attempt[key] for key in ("recipe", "seed", "run_id")
                )
                try:
                    run = self._report_only(run_id)
                    if attempt["status"] != "completed" or not run["eligible"]:
                        raise ValueError("required run not completed")
                    if (
                        run["source"] != config["source"]
                        or run["inputs"] != config["protocol"]["inputs"]
                    ):
                        raise ValueError(
                            "run source/input identity differs from fixed study"
                        )
                    path = self.path / "runs" / _identifier(run_id) / "predictions.csv"
                    sha256 = run["evidence"]["files"]["predictions.csv"]
                    runs[recipe][seed] = pd.read_csv(
                        BytesIO(checked_bytes(path, sha256))
                    )
                    validate_prediction_rows(runs[recipe][seed], rows)
                    evidence[recipe][seed] = _validation_run_evidence(run_id, run)
                except (ValueError, FileNotFoundError, KeyError) as error:
                    runs[recipe].pop(seed, None)
                    unavailable[run_id] = f"{type(error).__name__}: {error}"
            stored, stored_identity = None, None
            if stored_predictions is not None:
                assert stored_sidecar is not None  # Resolved together above.
                try:
                    stored, stored_identity = _read_predictions(
                        stored_predictions,
                        stored_sidecar,
                        rows,
                        config["acceptance"]["historical_checkpoint_sha256"],
                        split_record(inputs.split_dir),
                    )
                    stored_identity["checkpoint_sha256"] = config["acceptance"][
                        "historical_checkpoint_sha256"
                    ]
                except (ValueError, FileNotFoundError, KeyError) as error:
                    unavailable["stored_v2"] = f"{type(error).__name__}: {error}"
            else:
                unavailable["stored_v2"] = (
                    "stored v2 validation predictions not supplied"
                )
            report = build_validation_report(runs, stored, bootstrap=False)
            report.update(
                execution_mode=config["mode"],
                run_evidence=evidence,
                unavailable_evidence=unavailable,
                identity={
                    "protocol_sha256": file_sha256(self.path / "protocol.json"),
                    "execution_config_sha256": file_sha256(
                        self.path / "execution-config.json"
                    ),
                    "source": config["source"],
                    "environment": config["environment"],
                    "inputs": config["protocol"]["inputs"],
                    "stored_v2": stored_identity,
                },
                reconstruction={
                    "source": source_record(),
                    "environment": environment_record(),
                },
            )
        with self.evaluation_timing(bootstrap=True):
            report["bootstrap"] = bootstrap_intervals(runs, config["ci"])
        write_json(report, destination / "validation.json")
        write_json(
            {
                "file": "validation.json",
                "sha256": file_sha256(destination / "validation.json"),
            },
            destination / "checksum.json",
        )
        return report

    def freeze_selection(
        self,
        validation: Path | None = None,
        *,
        stored_checkpoint: Path | None = None,
    ) -> dict:
        """Freeze six completed runs and seed42 only after both validation gates pass."""
        from SpCas9.spcas9_finalist import freeze_selection

        if validation is None or stored_checkpoint is None:
            raise PermissionError(
                "finalist freeze requires validation evidence and stored checkpoint"
            )
        return freeze_selection(self, validation, stored_checkpoint)

    def evaluate_test(
        self,
        destination: Path | None = None,
        *,
        backend: "TrainingBackend | None" = None,
    ) -> dict:
        """Score exactly the frozen seven models; Test cannot reselect a recipe/seed."""
        from SpCas9.spcas9_finalist import evaluate_test

        return evaluate_test(self, destination, backend)

    def test_report(self, destination: Path) -> dict:
        """Verify saved Test results without inference or training."""
        from SpCas9.spcas9_finalist import test_report

        with self.evaluation_timing():
            return test_report(self, destination)

    def research_report(
        self,
        destination: Path,
        *,
        validation: Path | None = None,
        test: Path | None = None,
        scanner: Path | None = None,
    ) -> dict:
        """Report implementation, V2-A and independent V2-B without retraining."""
        from SpCas9.spcas9_research import research_report

        return research_report(
            self, destination, validation=validation, test=test, scanner=scanner
        )
