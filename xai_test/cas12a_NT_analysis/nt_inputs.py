"""Read-only source, membership, selected-export and actual DNA-token identity."""

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from cas12a.cnn_manifest import load_library, verify_against_library, verify_manifest
from cas12a.nt_data import digest

from .nt_run import file_hash

PRODUCTION_RECEIPT_SHA256 = (
    "d79ab828a2d4eccd97cc44c019d515d5b8c3e19ff5580850b0c313b05241f623"
)
EXPORT_FILES = {
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
}
LAYOUT = {
    "coordinate_system": "0-based half-open",
    "left_flank": [0, 4],
    "pam": [4, 8],
    "protospacer": [8, 31],
    "right_flank": [31, 34],
}
DNA_SPANS = [
    [0, 6],
    [6, 12],
    [12, 18],
    [18, 24],
    [24, 30],
    [30, 31],
    [31, 32],
    [32, 33],
    [33, 34],
]
MAX_LENGTH = 12


def check_export_files(export_dir, expected):
    export_dir = Path(export_dir)
    actual = {
        path.relative_to(export_dir).as_posix(): file_hash(path)
        for path in export_dir.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError("selected export hash/file set changed since verification")
    return actual


@dataclass(frozen=True)
class VerifiedInputs:
    rows: list
    identity: dict
    export_dir: Path
    fixture: bool


def export_identity(export_dir, receipt_path, *, fixture):
    export_dir, receipt_path = Path(export_dir), Path(receipt_path)
    if not fixture and file_hash(receipt_path) != PRODUCTION_RECEIPT_SHA256:
        raise ValueError("production export receipt identity mismatch")
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get("validated") is not True:
        raise ValueError("selected export receipt is not validated")
    actual = {
        p.relative_to(export_dir).as_posix(): file_hash(p)
        for p in export_dir.rglob("*")
        if p.is_file()
    }
    if not EXPORT_FILES <= actual.keys() or (
        not fixture and set(actual) != EXPORT_FILES
    ):
        raise ValueError("selected export file set mismatch")
    for name, checksum in actual.items():
        if receipt["files"].get(f"selected_export/{name}") != checksum:
            raise ValueError(f"selected export hash mismatch: {name}")
    return actual


def verify_inputs(source, manifest_path, export_dir, receipt_path, *, fixture=False):
    """Validate existing membership without constructing or writing any partition."""
    return _verify_partition(
        source, manifest_path, export_dir, receipt_path, "validation", fixture=fixture
    )


def verify_test_inputs(
    source, manifest_path, export_dir, receipt_path, *, fixture=False
):
    """Read the original Test membership, with a separate Test identity contract."""
    return _verify_partition(
        source, manifest_path, export_dir, receipt_path, "test", fixture=fixture
    )


def _verify_partition(
    source, manifest_path, export_dir, receipt_path, partition, *, fixture
):
    manifest = json.loads(Path(manifest_path).read_bytes())
    verify_manifest(manifest)
    if (manifest["run_kind"] == "fixture") != bool(fixture):
        raise ValueError("fixture/production input mismatch")
    data = load_library(source, manifest)
    by_id = verify_against_library(manifest, data)
    ids = sorted(manifest["partitions"][partition])
    if partition == "test" and not fixture and len(ids) != 1500:
        raise ValueError("production Cas12a Test membership must contain 1,500 rows")
    files = export_identity(export_dir, receipt_path, fixture=fixture)
    identity = {
        "fixture": bool(fixture),
        "source_sha256": data["source_file_sha256"],
        "table_sha256": data["table_sha256"],
        "partition_sha256": manifest["partition_sha256"],
        "manifest_sha256": digest(manifest),
        "export_files": files,
        "receipt_sha256": file_hash(receipt_path),
        "partition": partition,
        f"{partition}_source_rows": ids,
        f"{partition}_rows": [row_identity(by_id[i]) for i in ids],
    }
    return VerifiedInputs(
        [by_id[i] for i in ids],
        identity,
        Path(export_dir).resolve(),
        bool(fixture),
    )


def validate_vocabulary(tokenizer, embedding_rows):
    vocab = tokenizer.get_vocab()
    if embedding_rows != 4105 or len(vocab) != 4107 or len(set(vocab.values())) != 4107:
        raise ValueError("expected 4107 vocabulary IDs and 4105 embedding rows")
    outside = {
        token: index
        for token, index in vocab.items()
        if index < 0 or index >= embedding_rows
    }
    if set(outside) != {"<eos>", "<bos>"} or set(outside.values()) != {4105, 4106}:
        raise ValueError("only unused EOS/BOS may exceed embedding rows")
    dna = ["".join(bases) for bases in itertools.product("ACGT", repeat=6)] + list(
        "ACGT"
    )
    if any(
        token not in vocab or not 0 <= vocab[token] < embedding_rows for token in dna
    ):
        raise ValueError("DNA vocabulary is not supported by actual embeddings")
    if (tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id) != (
        3,
        1,
        0,
    ):
        raise ValueError("saved special-token identity mismatch")
    return {
        "vocabulary_sha256": digest(vocab),
        "vocabulary_size": len(vocab),
        "embedding_rows": embedding_rows,
    }


def encode_sequences(tokenizer, sequences, embedding_rows):
    """Check exact executed CLS/DNA/padding IDs, masks, spans and reconstruction."""
    records = []
    for sequence in sequences:
        if (
            not isinstance(sequence, str)
            or len(sequence) != 34
            or set(sequence) - set("ACGT")
        ):
            raise ValueError("expected ACGT 34bp sequence")
        encoded = tokenizer(
            sequence,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=False,
            return_attention_mask=True,
        )
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        expected_tokens = [sequence[start:end] for start, end in DNA_SPANS]
        dna_ids = tokenizer.convert_tokens_to_ids(expected_tokens)
        expected_ids = [
            tokenizer.cls_token_id,
            *dna_ids,
            tokenizer.pad_token_id,
            tokenizer.pad_token_id,
        ]
        if ids != expected_ids or mask != [1] * 10 + [0, 0]:
            raise ValueError("lossless actual token span/special/padding mismatch")
        if (
            any(
                type(index) is not int or not 0 <= index < embedding_rows
                for index in ids
            )
            or tokenizer.unk_token_id in ids
        ):
            raise ValueError("executed token ID exceeds embedding or is UNK")
        recovered = "".join(tokenizer.convert_ids_to_tokens(ids[1:10]))
        unpadded = tokenizer(sequence, padding=False, truncation=False)["input_ids"]
        if (
            recovered != sequence
            or unpadded != ids[:10]
            or tokenizer(recovered, padding=False, truncation=False)["input_ids"]
            != unpadded
        ):
            raise ValueError("retokenization is not lossless")
        records.append(
            {
                "input_ids": ids,
                "attention_mask": mask,
                "special_mask": [True] + [False] * 9 + [True, True],
                "dna_spans": DNA_SPANS,
            }
        )
    return records


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def row_identity(row):
    return {
        "source_row": row["source_row"],
        "sequence_sha256": row["sequence_sha256"],
        "label": row["target"],
        "source_identity_sha256": digest(row["raw"]),
    }
