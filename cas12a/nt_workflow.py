"""Public NT-Cas12a data preparation, runtime preflight and fit initialization.

This module does not import a training runtime. Prepared runs are not trained
models. Heavy dependencies load only when runtime operations are requested.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from cas12a.nt_data import (
    SERIALIZATION,
    SplitConfig,
    canonical_bytes,
    digest,
    partition_library,
    read_library,
    save_or_verify_partition,
)

__all__ = ["FitFixture", "ModelSource", "NTWorkflow", "PreparedRun", "SplitConfig"]


@dataclass(frozen=True)
class FitFixture:
    """Explicit offline controls; production never accepts shortened intervals.

    Optional monitor values replace only the selection monitor, while real
    validation predictions and measured losses are still checked and recorded.
    """

    interval_updates: int = 1
    validation_losses: tuple | None = None

    def __post_init__(self):
        import math

        if (
            type(self.interval_updates) is not int
            or not 1 <= self.interval_updates <= 125
        ):
            raise ValueError("fixture interval must be an integer from 1 to 125")
        if self.validation_losses is not None:
            values = tuple(self.validation_losses)
            if not values or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(
                    "fixture validation losses must be finite and nonnegative"
                )
            object.__setattr__(self, "validation_losses", values)


@dataclass(frozen=True)
class ModelSource:
    """Production always resolves the original backbone to an immutable commit."""

    revision: str = "main"
    fixture_dir: Path | None = None

    @classmethod
    def fixture(cls, directory):
        return cls(fixture_dir=Path(directory).resolve())


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    local_dir: Path
    durable_dir: Path
    status: dict

    @property
    def checkpoint_dir(self):
        return self.local_dir / "checkpoints"


@dataclass(frozen=True)
class NTWorkflow:
    source: Path
    partition_path: Path
    local_work_root: Path
    drive_output_root: Path
    split: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self):
        for name in (
            "source",
            "partition_path",
            "local_work_root",
            "drive_output_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        local, durable = self.local_work_root, self.drive_output_root
        if local.is_relative_to(durable) or durable.is_relative_to(local):
            raise ValueError(
                "local work and durable output roots must be separate, non-overlapping locations"
            )

    def prepare(self, *, run_id=None):
        """Validate the source, verify/create membership, and reserve a new run.

        Existing membership is compared with the complete deterministic contract
        before reuse. No existing run destination or partition is overwritten.
        """
        run_id = run_id if run_id is not None else uuid4().hex
        if not isinstance(run_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id
        ):
            raise ValueError(
                "run_id must be 1-80 ASCII letters/digits/underscores/hyphens"
            )
        local_dir = self.local_work_root / run_id
        durable_dir = self.drive_output_root / run_id
        for directory in (local_dir, durable_dir):
            if directory.exists():
                raise FileExistsError(f"run destination already exists: {directory}")
            if self.partition_path.is_relative_to(
                directory
            ) or self.source.is_relative_to(directory):
                raise ValueError(
                    "source and reusable partition must be outside the new run destinations"
                )
        data = read_library(self.source, self.split)
        partition = save_or_verify_partition(
            self.partition_path, partition_library(data, self.split)
        )
        configuration = {
            "serialization": SERIALIZATION,
            "run_id": run_id,
            "run_kind": self.split.run_kind,
            "source": str(self.source),
            "partition_path": str(self.partition_path),
            "local_work_root": str(self.local_work_root),
            "drive_output_root": str(self.drive_output_root),
            "checkpoint_dir": str(local_dir / "checkpoints"),
            "split": {
                "counts": partition["counts"],
                "seed": partition["seed"],
                "algorithm": partition["algorithm"],
            },
            "data_sha256": digest(data),
            "partition_manifest_sha256": digest(partition),
        }
        status = {
            "serialization": SERIALIZATION,
            "run_id": run_id,
            "run_kind": self.split.run_kind,
            "state": "preparing",
            "model_completed": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256": digest(configuration),
        }
        # mkdir without exist_ok is the final, race-safe reservation check.
        local_dir.mkdir(parents=True)
        reserved = [local_dir]
        try:
            durable_dir.mkdir(parents=True)
            reserved.append(durable_dir)
            (local_dir / "checkpoints").mkdir()
            for directory in (local_dir, durable_dir):
                (directory / "status.json").write_bytes(canonical_bytes(status))
            for filename, value in (
                ("data.json", data),
                ("partition.json", partition),
                ("configuration.json", configuration),
            ):
                encoded = canonical_bytes(value)
                for directory in (local_dir, durable_dir):
                    with (directory / filename).open("xb") as handle:
                        handle.write(encoded)
                    if (directory / filename).read_bytes() != encoded:
                        raise OSError(
                            f"artifact verification failed: {directory / filename}"
                        )
            status["state"] = "prepared"
            for directory in (local_dir, durable_dir):
                (directory / "status.json").write_bytes(canonical_bytes(status))
                if (directory / "status.json").read_bytes() != canonical_bytes(status):
                    raise OSError(f"status verification failed: {directory}")
        except Exception as error:
            status.update(
                state="preparation_failed", error=f"{type(error).__name__}: {error}"
            )
            # Only touch destinations reserved by this call. If storage is
            # unavailable, report what survived without masking the cause.
            for directory in reserved:
                try:
                    (directory / "status.json").write_bytes(canonical_bytes(status))
                except OSError as status_error:
                    error.add_note(
                        f"Could not record failure in {directory}: {status_error}"
                    )
            raise
        return PreparedRun(run_id, local_dir, durable_dir, status)

    def _read_evidence(self, run, filenames):
        documents = {}
        for filename in filenames:
            local = (run.local_dir / filename).read_bytes()
            if local != (run.durable_dir / filename).read_bytes():
                raise ValueError(f"run artifact identity differs: {filename}")
            documents[filename] = json.loads(local)
        return documents

    def _verified_run(self, run, allowed_states):
        if (
            run.local_dir != self.local_work_root / run.run_id
            or run.durable_dir != self.drive_output_root / run.run_id
        ):
            raise ValueError("run does not belong to this workflow")
        documents = self._read_evidence(
            run,
            (
                "configuration.json",
                "data.json",
                "partition.json",
                "status.json",
            ),
        )
        configuration = documents["configuration.json"]
        status = documents["status.json"]
        if status["state"] not in allowed_states or status[
            "configuration_sha256"
        ] != digest(configuration):
            raise ValueError(
                "run state/configuration is incompatible with this operation"
            )
        data = read_library(self.source, self.split)
        partition = save_or_verify_partition(
            self.partition_path, partition_library(data, self.split)
        )
        if (
            digest(data) != configuration["data_sha256"]
            or digest(partition) != configuration["partition_manifest_sha256"]
            or documents["data.json"] != data
            or documents["partition.json"] != partition
            or configuration["run_kind"] != self.split.run_kind
        ):
            raise ValueError("prepared data/partition identity changed")
        return data, partition, status

    def _record(self, run, filename, value):
        encoded = canonical_bytes(value)
        for directory in (run.local_dir, run.durable_dir):
            (directory / filename).write_bytes(encoded)
            if (directory / filename).read_bytes() != encoded:
                raise OSError(f"artifact verification failed: {directory / filename}")

    def preflight(self, run, *, model=None, device=None):
        """Validate all input tokens and measure two training updates + validation.

        CPU execution is restricted to explicitly marked fixture runs. A passed
        preflight freezes settings; it never marks a model as trained.
        """
        data, partition, status = self._verified_run(run, {"prepared"})
        evidence = {
            "attempts": [],
            "gpu_preflight_executed": False,
            "evidence_kind": "unexecuted",
            "training_source_rows": partition["partitions"]["train"],
            "validation_source_rows": partition["partitions"]["validation"],
        }
        try:
            status.update(state="preflighting")
            self._record(run, "status.json", status)
            from cas12a import nt_runtime as runtime

            source = runtime.resolve_source(model or ModelSource(), self.split.run_kind)
            evidence["source"] = source
            tokenizer = runtime.load_tokenizer(source)
            encoded, tokenization = runtime.tokenize_library(tokenizer, data["rows"])
            self._record(run, "tokenization.json", tokenization)
            environment = runtime.runtime_environment(device, self.split.run_kind)
            evidence["evidence_kind"] = (
                (
                    "gpu_fixture"
                    if environment["device"] == "cuda"
                    else "offline_fixture"
                )
                if self.split.is_fixture
                else "production_gpu"
            )
            evidence["gpu_preflight_executed"] = environment["device"] == "cuda"
            evidence["environment"] = environment
            settings, model_metadata = runtime.probe_candidates(
                source,
                environment,
                encoded,
                data["rows"],
                partition["partitions"],
                evidence,
            )
            execution = {
                "frozen": True,
                "source": source,
                "settings": settings,
                "environment": environment,
                "model": model_metadata,
                "initialization_sha256": evidence["attempts"][-1][
                    "initialization_sha256"
                ],
                "configuration_sha256": status["configuration_sha256"],
                "tokenization_sha256": digest(tokenization),
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "foreach": False,
                },
                "schedule": {
                    "name": "linear",
                    "warmup_ratio": 0.05,
                    "max_epochs": 10,
                    "seed": 42,
                    "max_grad_norm": 1.0,
                },
            }
            execution["optimizer"]["resolved_defaults"] = evidence["attempts"][-1][
                "optimizer_defaults"
            ]
            execution["schedule"].update(evidence["attempts"][-1]["scheduler"])
            execution["gradient_scaling"] = evidence["attempts"][-1]["gradient_scaling"]
            self._record(run, "preflight.json", evidence)
            execution["preflight_sha256"] = digest(evidence)
            self._record(run, "execution.json", execution)
            status.update(
                state="preflight_passed",
                execution_sha256=digest(execution),
                model_completed=False,
            )
            self._record(run, "status.json", status)
        except (Exception, KeyboardInterrupt) as error:
            status.update(
                state="preflight_failed",
                model_completed=False,
                error=f"{type(error).__name__}: {error}",
            )
            for filename, value in (
                ("preflight.json", evidence),
                ("status.json", status),
            ):
                for directory in (run.local_dir, run.durable_dir):
                    try:
                        (directory / filename).write_bytes(canonical_bytes(value))
                    except OSError as storage_error:
                        error.add_note(
                            f"Could not record {filename} in {directory}: {storage_error}"
                        )
            raise
        return PreparedRun(run.run_id, run.local_dir, run.durable_dir, status)

    def initialize_fit(self, run):
        """Create unstepped generic state using the verified frozen preflight.

        This is initialization only. The later fit operation owns training
        progress; preflight status never implies completed model training.
        """
        data, partition, status = self._verified_run(run, {"preflight_passed"})
        documents = self._read_evidence(
            run, ("execution.json", "preflight.json", "tokenization.json")
        )
        execution = documents["execution.json"]
        if (
            digest(execution) != status["execution_sha256"]
            or execution["configuration_sha256"] != status["configuration_sha256"]
            or digest(documents["preflight.json"]) != execution["preflight_sha256"]
            or digest(documents["tokenization.json"])
            != execution["tokenization_sha256"]
            or not execution["frozen"]
        ):
            raise ValueError("frozen preflight/configuration identity changed")
        from cas12a import nt_runtime as runtime

        environment = runtime.runtime_environment(
            execution["settings"]["device"], self.split.run_kind
        )
        if environment != execution["environment"]:
            raise ValueError(
                "runtime/dependency/GPU configuration changed since preflight"
            )
        tokenizer = runtime.load_tokenizer(execution["source"])
        _, tokenization = runtime.tokenize_library(tokenizer, data["rows"])
        if digest(tokenization) != execution["tokenization_sha256"]:
            raise ValueError("tokenizer identity changed since preflight")
        state = runtime.initialize(
            execution["source"],
            execution["settings"],
            train_rows=len(partition["partitions"]["train"]),
            tokenizer=tokenizer,
        )
        if state.initialization_sha256 != execution["initialization_sha256"]:
            raise ValueError("generic initialization differs from preflight")
        state.execution = execution
        return state

    def _runtime_failure(self, run, status, error, state):
        status.update(
            state=state,
            model_completed=False,
            error=f"{type(error).__name__}: {error}",
        )
        for directory in (run.local_dir, run.durable_dir):
            try:
                (directory / "status.json").write_bytes(canonical_bytes(status))
            except OSError as storage_error:
                error.add_note(
                    f"Could not record failure in {directory}: {storage_error}"
                )

    def fit(self, run, *, fixture=None):
        """Train the fixed recipe and retain the best/latest local resume states.

        No state from preflight enters fit. A run can be fitted only once; local
        resume is a separate lifecycle operation, not an implicit fresh fit.
        """
        if fixture is not None and (
            not self.split.is_fixture or not isinstance(fixture, FitFixture)
        ):
            raise ValueError(
                "fit fixture controls require an explicitly marked fixture run"
            )
        data, partition, status = self._verified_run(run, {"preflight_passed"})
        try:
            state = self.initialize_fit(run)
            from cas12a import nt_training

            status.update(state="training", model_completed=False)
            self._record(run, "status.json", status)
            history = nt_training.fit(
                state,
                data,
                partition,
                run,
                fixture,
                record=lambda name, value: self._record(run, name, value),
            )
            status.update(
                state="training_completed",
                history_sha256=digest(history),
                selected_global_step=history["selected_global_step"],
                selected_checkpoint_sha256=history["selected_checkpoint_sha256"],
            )
            self._record(run, "status.json", status)
        except (Exception, KeyboardInterrupt) as error:
            self._runtime_failure(
                run,
                status,
                error,
                "training_interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "training_failed",
            )
            raise
        return PreparedRun(run.run_id, run.local_dir, run.durable_dir, status)

    def export_selected(self, run):
        """Export the selected local checkpoint and verify a fresh model/tokenizer.

        Large artifacts stay local. Validated local export does not establish
        durable export or completed final-Testset evaluation.
        """
        data, partition, status = self._verified_run(run, {"training_completed"})
        try:
            from cas12a import nt_training

            documents = self._read_evidence(
                run, ("execution.json", "history.json", "training.json")
            )
            if (
                digest(documents["execution.json"]) != status["execution_sha256"]
                or digest(documents["history.json"]) != status["history_sha256"]
                or digest(documents["training.json"])
                != documents["history.json"]["training_sha256"]
            ):
                raise ValueError("training/export evidence identity changed")
            status.update(state="exporting", local_export_validated=False)
            self._record(run, "status.json", status)
            report = nt_training.export_selected(
                run,
                data,
                partition,
                documents["execution.json"],
                documents["history.json"],
            )
            self._record(run, "export.json", report)
            status.update(
                state="export_validated",
                local_export_validated=True,
                durable_export_validated=False,
                export_sha256=digest(report),
            )
            self._record(run, "status.json", status)
        except (Exception, KeyboardInterrupt) as error:
            status["local_export_validated"] = False
            self._runtime_failure(run, status, error, "export_failed")
            raise
        return PreparedRun(run.run_id, run.local_dir, run.durable_dir, status)

    def verify_export(self, run):
        """Recheck saved identities and reload before consuming a selected export."""
        data, partition, status = self._verified_run(run, {"export_validated"})
        try:
            from cas12a import nt_training

            documents = self._read_evidence(
                run, ("execution.json", "history.json", "export.json")
            )
            if (
                digest(documents["execution.json"]) != status["execution_sha256"]
                or digest(documents["history.json"]) != status["history_sha256"]
                or digest(documents["export.json"]) != status["export_sha256"]
            ):
                raise ValueError("selected export evidence identity changed")
            nt_training.verify_export(
                run,
                data,
                partition,
                documents["execution.json"],
                documents["history.json"],
                documents["export.json"],
            )
        except (Exception, KeyboardInterrupt) as error:
            # Revoke a previous success if either evidence or artifact changed.
            try:
                try:
                    report = json.loads((run.local_dir / "export.json").read_bytes())
                except (OSError, ValueError):
                    report = {}
                if not isinstance(report, dict):
                    report = {}
                report.update(validated=False, error=f"{type(error).__name__}: {error}")
                self._record(run, "export.json", report)
            except (OSError, ValueError) as storage_error:
                error.add_note(f"Could not record export invalidation: {storage_error}")
            status["local_export_validated"] = False
            self._runtime_failure(run, status, error, "export_failed")
            raise
        return PreparedRun(run.run_id, run.local_dir, run.durable_dir, status)
