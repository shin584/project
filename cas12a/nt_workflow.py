"""Public NT-Cas12a lifecycle interface, beginning with data/run preparation.

This module does not import a training runtime. Prepared runs are not trained
models. Later lifecycle operations belong on this same workflow boundary.
"""

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

__all__ = ["NTWorkflow", "PreparedRun", "SplitConfig"]


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
