"""Read pinned study inputs without repairing or regenerating a partition."""

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from SpCas9.spcas9_data import DATASET_PATH, HEAD_NAMES, SEQUENCE_COLUMN
from SpCas9.spcas9_eval import METADATA_COLUMNS
from SpCas9.spcas9_split import CHECKSUM_FILENAME, SPLIT_FILENAMES
from SpCas9.spcas9_split import OUTPUT_DIR as SPLIT_DIR

REQUIRED_ROWS = {"train": 29134, "val": 3676, "test": 5186}


@dataclass(frozen=True)
class InputPins:
    dataset_sha256: str
    sidecar_sha256: str
    files: dict[str, str]


# Independent pins: never derive the expected identity from the inputs being checked.
PUBLISHED_PINS = InputPins(
    "0c570ea6e85221654cf9d2130356b4e87f978942e9cbd1184c4efccaac77572e",
    "49be8031659772166460f07d103a6966550fa6f05e2d30b52102cc90a2d6cc3a",
    {
        "train.csv": "4c1ffc29c4db87cab5e5da1d29abe3a84440050c66b1a6c3b30daaf723812c01",
        "val.csv": "0757d7c2f7766e13d4c9a4d62fd1b6570ca17076087b711c139ec1ff6b54728d",
        "test.csv": "1a1305b6e0eeef91625443f1e21f85d233f8e0dac9e8be3cdb3daea5eeaf549b",
    },
)


def checked_bytes(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 {actual} differs from pinned {expected}")
    return raw


@dataclass(frozen=True)
class StudyInputs:
    source_path: Path = DATASET_PATH
    split_dir: Path = SPLIT_DIR
    pins: InputPins = PUBLISHED_PINS

    def record(self) -> dict:
        return {
            "source_path": str(self.source_path.resolve()),
            "split_dir": str(self.split_dir.resolve()),
            "dataset_sha256": self.pins.dataset_sha256,
            "sidecar_sha256": self.pins.sidecar_sha256,
            "files": dict(self.pins.files),
        }

    @classmethod
    def from_record(cls, record: dict) -> "StudyInputs":
        return cls(
            Path(record["source_path"]),
            Path(record["split_dir"]),
            InputPins(
                record["dataset_sha256"], record["sidecar_sha256"], record["files"]
            ),
        )

    def validate(self) -> dict[str, pd.DataFrame]:
        """Hash and parse the same bytes; Test is inspected only for integrity."""
        checked_bytes(self.source_path, self.pins.dataset_sha256)
        sidecar = json.loads(
            checked_bytes(self.split_dir / CHECKSUM_FILENAME, self.pins.sidecar_sha256)
        )
        if sidecar["dataset_sha256"] != self.pins.dataset_sha256:
            raise ValueError("sidecar source SHA-256 differs from pinned source")
        frames = {}
        for name, count in REQUIRED_ROWS.items():
            filename = SPLIT_FILENAMES[name]
            if sidecar["files"][filename] != self.pins.files[filename]:
                raise ValueError(f"sidecar {filename} SHA-256 differs from pinned CSV")
            raw = checked_bytes(self.split_dir / filename, self.pins.files[filename])
            frame = pd.read_csv(BytesIO(raw))
            if len(frame) != count or sidecar["rows"][name] != count:
                raise ValueError(f"{name}: requires exactly {count} rows")
            if not set(METADATA_COLUMNS + list(HEAD_NAMES)) <= set(frame.columns):
                raise ValueError(f"{name}: missing split columns")
            frame[SEQUENCE_COLUMN] = (
                frame[SEQUENCE_COLUMN].astype(str).str.strip().str.upper()
            )
            if not frame[SEQUENCE_COLUMN].str.len().eq(30).all():
                raise ValueError(f"{name}: requires 30bp sequences")
            if not frame[SEQUENCE_COLUMN].str.fullmatch("[ACGT]{30}").all():
                raise ValueError(f"{name}: requires A/C/G/T sequences")
            if (
                not frame["protospacer"]
                .eq(frame[SEQUENCE_COLUMN].str.slice(4, 24))
                .all()
            ):
                raise ValueError(f"{name}: protospacer differs from consumed sequence")
            if (
                not frame["split"].eq(name).all()
                or frame["row_index"].duplicated().any()
            ):
                raise ValueError(f"{name}: invalid split-row identity")
            values = frame[list(HEAD_NAMES)].apply(pd.to_numeric, errors="raise")
            if (
                np.isinf(values.to_numpy(dtype=float)).any()
                or not values.notna().any(axis=1).all()
            ):
                raise ValueError(f"{name}: invalid measured labels")
            frame[list(HEAD_NAMES)] = values
            frames[name] = frame
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            for column in ("row_index", SEQUENCE_COLUMN, "protospacer"):
                if set(frames[left][column]) & set(frames[right][column]):
                    raise ValueError(f"{left}/{right}: {column} overlap")
        return frames
