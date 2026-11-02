"""Explicit, inference-only 9-Hydra v2 export (#96).

The destination is a fresh directory containing model.keras and manifest.json.
Only manifest.json with validation.status == 'passed', together with the matching
artifact SHA-256, denotes success. Interrupted destinations must not be reused.
No spreadsheet, split, training module or evaluation report is imported here.
"""

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

HEADS = (
    ("SpCas9(WT)", "Output_SpCas9"),
    ("SpCas9-NG", "Output_SpCas9_NG"),
    ("VRQR", "Output_VRQR_variant"),
    ("xCas9", "Output_xCas"),
    ("Sniper", "Output_Sniper_Cas9"),
    ("SpCas9-HF1", "Output_SpCas9_HF_1"),
    ("eSpCas9(1.1)", "Output_eSpCas9_1_1_"),
    ("HypaCas9", "Output_HypaCas9"),
    ("evoCas9", "Output_evoCas9"),
)
RTOL = 1e-5
ATOL = 1e-6
# Four flank bases + twenty protospacer bases + three PAM + three flank.
FIXTURE_ID = "9hydra-complete-acgt-windows-v1"
WINDOWS = (
    "ACGT" + "ACGT" * 5 + "AGG" + "TCA",
    "TGCA" + "TGCA" * 5 + "TGG" + "ATC",
    "CATG" + "GATTACAGATTACAGATTAC" + "CGA" + "GTC",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(path):
    stat = path.stat()
    return {
        "sha256": _sha256(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _check_input(model):
    if len(model.inputs) != 1:
        raise ValueError("Input contract: expected exactly one In_Seq input")
    tensor = model.inputs[0]
    if (
        tensor.name.split(":")[0] != "In_Seq"
        or tuple(tensor.shape) != (None, 30, 4)
        or str(tensor.dtype) != "float32"
    ):
        raise ValueError(
            f"Input contract: expected In_Seq float32 (None, 30, 4); "
            f"got {tensor.name} {tensor.dtype} {tensor.shape}"
        )


def _named_tensors(model):
    _check_input(model)
    # Include nested models when detecting ambiguous identities; never infer
    # consumer order from a dict or from Keras' output_names.
    layers = []
    seen = set()

    def visit(container):
        for layer in container.layers:
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            layers.append(layer)
            if isinstance(layer, keras.Model):
                visit(layer)

    visit(model)
    tensors = []
    for _, name in HEADS:
        matches = [layer for layer in layers if layer.name == name]
        if len(matches) != 1:
            raise ValueError(
                f"Head {name}: expected unique layer, found {len(matches)}"
            )
        layer = matches[0]
        if len(layer._inbound_nodes) != 1:
            raise ValueError(f"Head {name}: ambiguous reused layer tensor")
        tensor = layer.output
        if (
            not hasattr(tensor, "shape")
            or tuple(tensor.shape) != (None, 1)
            or str(tensor.dtype) != "float32"
        ):
            raise ValueError(f"Head {name}: expected float32 (None, 1) tensor")
        activation = getattr(layer, "activation", None)
        if activation is None or keras.activations.serialize(activation) != "linear":
            raise ValueError(f"Head {name}: expected linear activation")
        tensors.append(tensor)
    return tensors


def _check_predictions(predictions, count, label):
    if type(predictions) is not list or len(predictions) != len(HEADS):
        raise ValueError(f"{label}: expected Python list of nine outputs")
    for (_, name), values in zip(HEADS, predictions, strict=True):
        if (
            not isinstance(values, np.ndarray)
            or values.shape != (count, 1)
            or values.dtype != np.float32
        ):
            raise ValueError(f"{label} {name}: expected float32 ({count}, 1) array")
        if not np.isfinite(values).all():
            raise ValueError(f"{label} {name}: nonfinite verification output")


def _verify(source_heads, reloaded):
    _named_tensors(reloaded)
    encoded = np.eye(4, dtype=np.float32)[
        [["ACGT".index(base) for base in window] for window in WINDOWS]
    ]
    batches = []
    for label, inputs in (("single", encoded[:1]), ("distinct_batch", encoded)):
        expected = source_heads.predict(inputs, verbose=0)
        actual = reloaded.predict(inputs, verbose=0)
        _check_predictions(expected, len(inputs), "source")
        _check_predictions(actual, len(inputs), "reload")
        errors = []
        for (protein, name), reference, observed in zip(
            HEADS, expected, actual, strict=True
        ):
            difference = np.abs(observed.astype(np.float64) - reference)
            try:
                np.testing.assert_allclose(observed, reference, rtol=RTOL, atol=ATOL)
            except AssertionError as exc:
                raise ValueError(f"Parity failed for {name} ({label}): {exc}") from exc
            errors.append(
                {
                    "protein": protein,
                    "layer": name,
                    "max_absolute_error": float(difference.max()),
                    "max_relative_error": float(
                        (
                            difference
                            / np.maximum(np.abs(reference), np.finfo(np.float32).tiny)
                        ).max()
                    ),
                    "source_values": reference[:, 0].tolist(),
                    "reloaded_values": observed[:, 0].tolist(),
                }
            )
        batches.append({"fixture": label, "n": len(inputs), "heads": errors})
    return {"status": "passed", "rtol": RTOL, "atol": ATOL, "batches": batches}


def export_checkpoint(source, destination):
    """Save/reload and verify an explicit checkpoint into a fresh directory.

    Return the verified manifest. Never overwrite source or destination. A failed
    publication can leave an incomplete destination without a success manifest;
    use a new destination for every retry.
    """
    source = Path(source).resolve(strict=True)
    requested_destination = Path(destination)
    destination = requested_destination.resolve()
    # is_symlink also catches a dangling alias, which exists() alone misses.
    if destination == source or (
        requested_destination.exists() and requested_destination.samefile(source)
    ):
        raise ValueError("Source/destination collision or alias")
    if requested_destination.exists() or requested_destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {requested_destination}")
    if not source.is_file():
        raise ValueError("Source must be a saved checkpoint file")
    if not destination.parent.is_dir():
        raise ValueError("Destination parent directory must already exist")
    identity = _source_identity(source)
    original = keras.models.load_model(source, compile=False)
    heads = _named_tensors(original)
    inference = keras.Model(inputs=original.inputs[0], outputs=heads)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.staging-", dir=destination.parent
    ) as staging_dir:
        staging = Path(staging_dir)
        artifact = staging / "model.keras"
        inference.save(artifact)
        output_sha256 = _sha256(artifact)
        reloaded = keras.models.load_model(artifact, compile=False)
        validation = _verify(inference, reloaded)
        if _sha256(artifact) != output_sha256:
            raise ValueError("Staged artifact identity changed during verification")
        if _source_identity(source) != identity:
            raise ValueError("Source identity changed during export")
        manifest = {
            "schema_version": 1,
            "source": {"path": str(source), **identity},
            "artifact": {"filename": "model.keras", "sha256": output_sha256},
            "input": {
                "name": "In_Seq",
                "dtype": "float32",
                "shape": [None, 30, 4],
                "channels": list("ACGT"),
            },
            "output": {
                "container": "list",
                "count": 9,
                "dtype": "float32",
                "shape_per_head": [None, 1],
                "activation": "linear",
                "scale": "published_percentage/100",
                "transforms": [],
            },
            "heads": [
                {"index": i, "protein": protein, "layer": name}
                for i, (protein, name) in enumerate(HEADS)
            ],
            "fixtures": {
                "id": FIXTURE_ID,
                "windows": list(WINDOWS),
                "sha256": hashlib.sha256("\n".join(WINDOWS).encode()).hexdigest(),
            },
            "validation": validation,
            "versions": {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "keras": keras.__version__,
                "numpy": np.__version__,
            },
            "verification_code": {
                "module": "SpCas9.spcas9_export",
                "sha256": _sha256(Path(__file__)),
            },
        }
        # Exclusive reservation prevents competing exports from overwriting each
        # other. Publish the marker last, only for the exact verified bytes.
        destination.mkdir()
        artifact.rename(destination / "model.keras")
        pending = destination / "manifest.pending"
        pending.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    # Cleanup is part of the fallible work, so it also precedes success.
    if _sha256(destination / "model.keras") != output_sha256:
        raise ValueError("Published artifact identity differs from verified artifact")
    if _source_identity(source) != identity:
        raise ValueError("Source identity changed before publication")
    pending.rename(destination / "manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        export_checkpoint(args.source, args.destination)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified export: {args.destination / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
