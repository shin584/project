"""Read-only 9-Hydra v2 diagnostic execution boundary (issue #69).

Runtime verification stages are deliberately blocked until their tickets supply
evidence. A completed harness execution is not verification of the predictor.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from SpCas9.spcas9_data import library_versions
from SpCas9.spcas9_execution import reserve_destination, source_record
from SpCas9.spcas9_split import file_sha256, write_json

ENVIRONMENT_SETTINGS = (
    "TF_ENABLE_ONEDNN_OPTS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TF_DETERMINISTIC_OPS",
    "TF_CUDNN_DETERMINISTIC",
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
    "KERAS_BACKEND",
    "CUBLAS_WORKSPACE_CONFIG",
)


class CheckState(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not applicable"


class Disposition(str, Enum):
    NO_DEFECT = "no defect established"
    EXPLAINED = "explained behavior"
    REPAIRED = "repaired defect"
    UNRESOLVED_DEFECT = "unresolved defect"
    MODEL_LIMITATION = "unresolved model limitation"
    EXCLUDED = "excluded scope"


@dataclass(frozen=True)
class ArtifactLocations:
    dataset: Path
    split_dir: Path
    checkpoint: Path
    training_log: Path
    predictions: Path
    prediction_sidecar: Path
    evidence: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "dataset",
            "split_dir",
            "checkpoint",
            "training_log",
            "predictions",
            "prediction_sidecar",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        object.__setattr__(
            self, "evidence", tuple(Path(p).resolve() for p in self.evidence)
        )

    def paths(self) -> tuple[Path, ...]:
        return tuple(
            Path(p).resolve()
            for p in (
                self.dataset,
                self.split_dir,
                self.checkpoint,
                self.training_log,
                self.predictions,
                self.prediction_sidecar,
                *self.evidence,
            )
        )


@dataclass(frozen=True)
class DiagnosticConfig:
    interpreter: Path
    device: str = "/CPU:0"
    inference_batch_size: int = 1024
    fixture_specification: str = (
        "Deferred controlled finite fixture: measured zero and nonzero; missing "
        "labels; fully masked head; differing head masks; nonzero residuals; "
        "uneven batches. Freeze exact values before runtime probes."
    )
    commands: tuple[str, ...] = field(
        default_factory=lambda: (subprocess.list2cmdline(sys.orig_argv),)
    )
    environment: tuple[tuple[str, str | None], ...] = field(
        default_factory=lambda: tuple(
            (key, os.environ.get(key)) for key in ENVIRONMENT_SETTINGS
        )
    )
    rtol: float = field(default=1e-5, init=False)
    atol: float = field(default=1e-6, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "interpreter", Path(self.interpreter).resolve())
        object.__setattr__(self, "commands", tuple(self.commands))
        object.__setattr__(
            self, "environment", tuple((k, v) for k, v in self.environment)
        )

    def validate(self) -> None:
        if not Path(self.interpreter).samefile(sys.executable):
            raise ValueError("run with the explicit configured interpreter")
        if type(self.inference_batch_size) is not int or self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be a positive integer")
        if (
            not isinstance(self.device, str)
            or not self.device
            or not isinstance(self.fixture_specification, str)
            or not self.fixture_specification
            or not self.commands
            or not all(isinstance(c, str) and c for c in self.commands)
        ):
            raise ValueError("device, fixture specification and commands are required")
        if self.rtol != 1e-5 or self.atol != 1e-6:
            raise ValueError("diagnostic tolerances are fixed at rtol=1e-5, atol=1e-6")
        if dict(self.environment) != {
            k: os.environ.get(k) for k in ENVIRONMENT_SETTINGS
        }:
            raise ValueError(
                "execution environment differs from the frozen configuration"
            )

    def record(self) -> dict:
        record = asdict(self)
        record["interpreter"] = str(Path(self.interpreter).resolve())
        record["environment"] = {
            key: {"value": value, "status": "unset/default" if value is None else "set"}
            for key, value in self.environment
        }
        return record


class EnvironmentBackend(Protocol):
    """Only environment observation; no model fitting or study capability."""

    def environment(self) -> dict: ...


class TensorFlowEnvironment:
    def environment(self) -> dict:
        import keras
        import tensorflow as tf

        return {
            "backend": keras.backend.backend(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "physical_devices": [d.name for d in tf.config.list_physical_devices()],
            "effective_onednn": "unknown (no runtime status observed)",
            "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            "thread_zero_meaning": "runtime default",
            "effective_determinism": "unknown",
        }


@dataclass(frozen=True)
class CheckResult:
    state: CheckState
    message: str
    disposition: Disposition = Disposition.NO_DEFECT
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Stage:
    name: str
    prerequisites: tuple[str, ...]
    execute: Callable[[], CheckResult]
    required: bool = True
    in_scope: bool = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _inference_only():
    """Reject accidental training before entering its body; restore profiling.

    This is a guard against adapter mistakes, not a sandbox for untrusted code.
    Stages execute synchronously in the calling thread.
    """
    previous = sys.getprofile()

    def guard(frame, event, arg):
        if event != "call":
            return
        name = frame.f_code.co_name
        module = frame.f_globals.get("__name__", "")
        forbidden = name in {
            "fit",
            "train_on_batch",
            "train_step",
            "apply_gradients",
            "update_step",
            "stateless_apply",
            "minimize",
            "_backend_apply_gradients",
        }
        filename = frame.f_code.co_filename.replace("\\", "/")
        forbidden |= filename.endswith("SpCas9/spcas9_train.py") and name in {
            "main",
            "train",
        }
        forbidden |= filename.endswith("SpCas9/spcas9_study.py") and name in {
            "plan",
            "plan_run",
            "train_validation",
            "freeze_selection",
            "evaluate_test",
        }
        if forbidden:
            raise RuntimeError(f"diagnostic operation forbidden: {module}.{name}")

    sys.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(previous)


def _deferred() -> CheckResult:
    return CheckResult(
        CheckState.BLOCKED, "Not implemented in #69; requires a later diagnostic ticket"
    )


def _stages(backend: EnvironmentBackend) -> tuple[_Stage, ...]:
    def environment() -> CheckResult:
        return CheckResult(
            CheckState.PASSED,
            "Backend environment observed; no model verified",
            evidence=backend.environment(),
        )

    return (
        _Stage("backend_environment", (), environment),
        _Stage("artifact_validation", ("input_snapshot",), _deferred),
        _Stage(
            "checkpoint_parity",
            ("artifact_validation", "backend_environment"),
            _deferred,
        ),
        _Stage("copy_reload", ("checkpoint_parity",), _deferred),
        _Stage("mask_metric_gradient", ("backend_environment",), _deferred),
        _Stage(
            "aggregation", ("artifact_validation", "backend_environment"), _deferred
        ),
        _Stage("finding_reports", ("artifact_validation",), _deferred),
    )


def _record(
    result: CheckResult, *, required: bool = True, in_scope: bool = True
) -> dict:
    return {
        **asdict(result),
        "state": CheckState(result.state),
        "disposition": Disposition(result.disposition),
        "required": required,
        "in_scope": in_scope,
        "finished_at": _now(),
    }


def _outcome(checks: dict) -> str:
    required = [c for c in checks.values() if c["required"] and c["in_scope"]]
    if any(c["state"] == CheckState.FAILED for c in required):
        return "failed"
    if not required or any(c["state"] != CheckState.PASSED for c in required):
        return "incomplete"
    return "verified"


def _snapshot(paths: tuple[Path, ...]) -> tuple[dict, list[str]]:
    files: dict = {}
    errors = []
    for path in paths:
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            members = sorted(path.rglob("*")) if path.is_dir() else [path]
            for member in members:
                if member.is_file():
                    files[str(member)] = {"before_sha256": file_sha256(member)}
        except OSError as error:
            errors.append(f"{type(error).__name__}: {error}")
    return files, errors


def run_diagnostics(
    artifacts: ArtifactLocations,
    destination: Path,
    config: DiagnosticConfig,
    *,
    backend: EnvironmentBackend | None = None,
) -> dict:
    """Reserve fresh evidence, freeze configuration and run independent checks.

    Missing inputs/backends and deferred required stages are incomplete, never
    verified. No source is replaced and failed runs retain their partial outputs.
    """
    config.validate()
    paths = artifacts.paths()
    destination = reserve_destination(destination, paths)
    started = _now()
    write_json(config.record(), destination / "configuration.json")
    inputs, errors = _snapshot(paths)
    checks = {
        "input_snapshot": _record(
            CheckResult(
                CheckState.BLOCKED if errors else CheckState.PASSED,
                "; ".join(errors)
                if errors
                else "Consumed input bytes hashed before execution",
            )
        )
    }
    try:
        source = source_record()
        source_result = CheckResult(
            CheckState.PASSED, "Revision and dirty source snapshot recorded"
        )
    except (OSError, subprocess.SubprocessError) as error:
        source = {"revision": None, "dirty": None, "files": {}, "error": str(error)}
        source_result = CheckResult(
            CheckState.BLOCKED, f"Source provenance unavailable: {error}"
        )
    checks["source_provenance"] = _record(source_result)
    manifest: dict = {
        "schema_version": 1,
        "started_at": started,
        "finished_at": None,
        "source": source,
        "inputs": inputs,
        "artifact_locations": [str(p) for p in paths],
        "environment": {
            "interpreter": str(Path(sys.executable).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "libraries": library_versions(),
            "backend": "unknown",
            "settings": config.record()["environment"],
            "device_requested": config.device,
            "effective_onednn": "unknown",
            "effective_determinism": "unknown",
        },
        "configuration": "configuration.json",
        "commands": list(config.commands),
        "outputs": {},
    }
    write_json(manifest, destination / "manifest.json")
    interrupted = False
    for stage in _stages(backend if backend is not None else TensorFlowEnvironment()):
        blocked = [
            name
            for name in stage.prerequisites
            if name not in checks or checks[name]["state"] != CheckState.PASSED
        ]
        if interrupted:
            result = CheckResult(
                CheckState.BLOCKED,
                "Execution interrupted; remaining stages not attempted",
            )
        elif blocked:
            result = CheckResult(
                CheckState.BLOCKED, f"Blocked by prerequisites: {', '.join(blocked)}"
            )
        else:
            try:
                with _inference_only():
                    result = stage.execute()
                # Validate strict JSON before persisting so an invalid stage
                # result cannot discard independent completed evidence.
                json.dumps(asdict(result), allow_nan=False)
                CheckState(result.state)
                Disposition(result.disposition)
            except KeyboardInterrupt:
                interrupted = True
                result = CheckResult(
                    CheckState.BLOCKED, "Execution interrupted during stage"
                )
            except (ImportError, OSError) as error:
                result = CheckResult(
                    CheckState.BLOCKED, f"{type(error).__name__}: {error}"
                )
            except Exception as error:  # noqa: BLE001 - retain independent diagnostic results
                result = CheckResult(
                    CheckState.FAILED, f"{type(error).__name__}: {error}"
                )
        checks[stage.name] = _record(
            result, required=stage.required, in_scope=stage.in_scope
        )
        write_json(
            {"outcome": "incomplete", "checks": checks}, destination / "report.json"
        )
    for path, entry in inputs.items():
        try:
            entry["after_sha256"] = file_sha256(Path(path))
            entry["unchanged"] = entry["before_sha256"] == entry["after_sha256"]
        except OSError as error:
            entry.update(after_sha256=None, unchanged=False, error=str(error))
    source_after: dict[str, str | None] = {}
    root = Path(__file__).resolve().parents[2]
    for name in source["files"]:
        try:
            source_after[name] = file_sha256(root / name)
        except OSError:
            source_after[name] = None
    source["after_files"] = source_after
    source["unchanged"] = (
        source["files"] == source_after if source["revision"] else None
    )
    preserved = (
        all(e["unchanged"] for e in inputs.values())
        and source["unchanged"] is not False
    )
    checks["source_preservation"] = _record(
        CheckResult(
            CheckState.PASSED if preserved else CheckState.FAILED,
            "Consumed source bytes unchanged"
            if preserved
            else "Source bytes changed or became unavailable",
            Disposition.NO_DEFECT if preserved else Disposition.UNRESOLVED_DEFECT,
        )
    )
    report = {
        "schema_version": 1,
        "issue": 69,
        "model": "9-Hydra v2",
        "outcome": _outcome(checks),
        "checks": checks,
        "started_at": started,
        "finished_at": _now(),
    }
    write_json(report, destination / "report.json")
    summary = f"9-Hydra v2 diagnostics: {report['outcome']}\n" + "".join(
        f"{name}: {c['state'].value} / {c['disposition'].value} - {c['message']}\n"
        for name, c in checks.items()
    )
    (destination / "summary.txt").write_text(summary, encoding="utf-8", newline="\n")
    manifest["finished_at"] = report["finished_at"]
    observed = checks["backend_environment"]
    manifest["environment"]["backend_observation"] = observed
    manifest["environment"]["backend"] = observed["evidence"].get("backend", "unknown")
    for key in (
        "effective_onednn",
        "effective_determinism",
        "physical_devices",
        "intra_op_threads",
        "inter_op_threads",
        "thread_zero_meaning",
    ):
        manifest["environment"][key] = observed["evidence"].get(key, "unknown")
    manifest["outputs"] = {
        p.relative_to(destination).as_posix(): file_sha256(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file() and p != destination / "manifest.json"
    }
    write_json(manifest, destination / "manifest.json")
    print(summary, end="")
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in (
        "dataset",
        "split-dir",
        "checkpoint",
        "training-log",
        "predictions",
        "prediction-sidecar",
        "destination",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--inference-batch-size", type=int, default=1024)
    args = parser.parse_args(argv)
    artifacts = ArtifactLocations(
        args.dataset,
        args.split_dir,
        args.checkpoint,
        args.training_log,
        args.predictions,
        args.prediction_sidecar,
        tuple(args.evidence),
    )
    config = DiagnosticConfig(
        Path(sys.executable),
        device=args.device,
        inference_batch_size=args.inference_batch_size,
        commands=(
            subprocess.list2cmdline(
                [
                    sys.executable,
                    "-m",
                    "SpCas9.spcas9_diagnostics",
                    *(sys.argv[1:] if argv is None else argv),
                ]
            ),
        ),
    )
    return run_diagnostics(artifacts, args.destination, config)


if __name__ == "__main__":
    raise SystemExit(0 if main()["outcome"] == "verified" else 1)
