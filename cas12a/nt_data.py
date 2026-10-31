"""Strict NT library and immutable partition evidence; NumPy/pandas only.

Serialization v1 is UTF-8 JSON, sorted keys, compact separators, no ASCII
escaping, no NaN/Infinity, and no trailing newline in hashed bytes. Source row
identity is the zero-based data-row position (the legacy loader's index).
"""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cas12a.data import SEQUENCE_COLUMN, TARGET_COLUMN
from cas12a.features import BASES, PAM_SLICE, SEQUENCE_LENGTH, protospacer_of
from cas12a.units import PERCENT_PER_UNIT, SCORE_MAX, SCORE_MIN

SERIALIZATION = "cas12a-nt-json-v1"
PARTITION_NAMES = ("train", "validation", "test")
PRODUCTION_COUNTS = (12000, 1500, 1500)
SEED = 42
ALGORITHM = "numpy.default_rng(42).permutation/source-row-ascending/v1"


def canonical_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SplitConfig:
    """Production counts are fixed; smaller synthetic runs require `fixture`."""

    counts: tuple = PRODUCTION_COUNTS
    is_fixture: bool = False

    def __post_init__(self):
        if (
            len(self.counts) != 3
            or any(type(count) is not int or count <= 0 for count in self.counts)
            or (not self.is_fixture and self.counts != PRODUCTION_COUNTS)
        ):
            raise ValueError(
                "split requires positive counts; production is 12000/1500/1500"
            )

    @classmethod
    def fixture(cls, train, validation, test):
        return cls(counts=(train, validation, test), is_fixture=True)

    @property
    def run_kind(self):
        return "fixture" if self.is_fixture else "production"


def _read_source(source_bytes, suffix):
    if suffix in (".csv", ".txt", ".tsv"):
        options = {"sep": "\t" if suffix == ".tsv" else ",", "keep_default_na": False}
        header = pd.read_csv(io.BytesIO(source_bytes), header=None, nrows=1, **options)
        frame = pd.read_csv(io.BytesIO(source_bytes), **options)
    elif suffix in (".xls", ".xlsx"):
        options = {"keep_default_na": False}
        header = pd.read_excel(
            io.BytesIO(source_bytes), header=None, nrows=1, **options
        )
        frame = pd.read_excel(io.BytesIO(source_bytes), **options)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    columns = header.iloc[0].tolist()
    if (
        any(not isinstance(column, str) or not column.strip() for column in columns)
        or len(set(columns)) != len(columns)
        or SEQUENCE_COLUMN not in columns
        or TARGET_COLUMN not in columns
        or list(frame.columns) != columns
        or not isinstance(frame.index, pd.RangeIndex)
    ):
        raise ValueError(
            "incompatible source schema: require unique named Cas12a columns"
        )
    return frame


def _missing(value):
    return pd.isna(value) or (isinstance(value, str) and not value.strip())


def _raw_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Spreadsheet metadata can contain dates; preserve an explicit type tag.
    if isinstance(value, (pd.Timestamp,)) or hasattr(value, "isoformat"):
        return {"datetime": value.isoformat()}
    raise ValueError(f"unsupported source cell type: {type(value).__name__}")


def read_library(source, split):
    """Validate raw sequences before any legacy sequence repair can run."""
    source = Path(source)
    source_bytes = source.read_bytes()
    frame = _read_source(source_bytes, source.suffix.lower())
    rows, excluded, source_rows = [], [], []
    seen_sequences, seen_protospacers = set(), set()
    for source_row, values in enumerate(frame.to_dict("records")):
        raw = {key: _raw_value(value) for key, value in values.items()}
        source_rows.append({"source_row": source_row, "raw": raw})
        sequence, label = values[SEQUENCE_COLUMN], values[TARGET_COLUMN]
        reasons = []
        if _missing(sequence):
            reasons.append("missing_sequence")
        target = None
        if _missing(label):
            reasons.append("missing_target")
        else:
            try:
                target = float(label)
            except (ValueError, TypeError):
                reasons.append("unparseable_target")
            else:
                if not np.isfinite(target):
                    raise ValueError(f"nonfinite target at source row {source_row}")
        if reasons:
            excluded.append({"source_row": source_row, "raw": raw, "reasons": reasons})
            continue
        sequence = str(sequence).strip().upper()
        if len(sequence) != SEQUENCE_LENGTH or set(sequence) - set(BASES):
            raise ValueError(f"expected 34 bp ACGT sequence at source row {source_row}")
        if sequence[PAM_SLICE][:3] != "TTT":
            raise ValueError(
                f"expected TTTN PAM at positions 4:8 at source row {source_row}"
            )
        protospacer = protospacer_of(sequence)
        if sequence in seen_sequences:
            raise ValueError(f"duplicate sequence at source row {source_row}")
        if protospacer in seen_protospacers:
            raise ValueError(f"duplicate protospacer at source row {source_row}")
        seen_sequences.add(sequence)
        seen_protospacers.add(protospacer)
        rows.append(
            {
                "source_row": source_row,
                "raw": raw,
                "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                "protospacer": protospacer,
                "target": float(
                    np.clip(target / PERCENT_PER_UNIT, SCORE_MIN, SCORE_MAX)
                ),
            }
        )
    if len(rows) != sum(split.counts):
        raise ValueError(
            f"usable row count: expected {sum(split.counts)}, got {len(rows)}"
        )
    canonical_table = {
        "serialization": SERIALIZATION,
        "columns": list(frame.columns),
        "rows": source_rows,
    }
    return {
        "serialization": SERIALIZATION,
        "source_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "table_sha256": digest(canonical_table),
        "columns": list(frame.columns),
        "source_row_count": len(frame),
        "rows": rows,
        "excluded": excluded,
    }


def partition_library(data, split):
    """Build the pinned partition and its training-only frozen baseline."""
    rows = sorted(data["rows"], key=lambda row: row["source_row"])
    permutation = np.random.default_rng(SEED).permutation(len(rows))
    partitions = {}
    offset = 0
    for name, count in zip(PARTITION_NAMES, split.counts):
        partitions[name] = [
            rows[int(i)]["source_row"] for i in permutation[offset : offset + count]
        ]
        offset += count
    # Full coverage plus globally unique sequences/protospacers checked above
    # establishes disjointness both within and across all three partitions.
    identities = [identity for members in partitions.values() for identity in members]
    if len(set(identities)) != len(rows) or set(identities) != {
        row["source_row"] for row in rows
    }:
        raise ValueError("partition must cover every usable source row exactly once")
    by_id = {row["source_row"]: row for row in rows}
    row_identities = [
        {"source_row": row["source_row"], "sequence_sha256": row["sequence_sha256"]}
        for row in rows
    ]
    training_identity = digest([by_id[i] for i in partitions["train"]])
    return {
        "serialization": SERIALIZATION,
        "run_kind": split.run_kind,
        "algorithm": ALGORITHM,
        "seed": SEED,
        "counts": dict(zip(PARTITION_NAMES, split.counts)),
        "source_file_sha256": data["source_file_sha256"],
        "table_sha256": data["table_sha256"],
        "rows": row_identities,
        "partitions": partitions,
        "partition_sha256": digest(partitions),
        "training_partition_sha256": training_identity,
        "baseline_train_mean": {
            "value": float(np.mean([by_id[i]["target"] for i in partitions["train"]])),
            "training_rows": len(partitions["train"]),
            "training_partition_sha256": training_identity,
        },
    }


def save_or_verify_partition(path, expected):
    """Never replace an existing membership, even if it is damaged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_bytes(expected))
        return expected
    except FileExistsError:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            raise ValueError(
                "partition manifest is unreadable; refusing to resplit"
            ) from error
        if canonical_bytes(saved) != canonical_bytes(expected):
            raise ValueError(
                "partition manifest differs from source/table/membership/configuration; refusing to resplit"
            )
        return saved
