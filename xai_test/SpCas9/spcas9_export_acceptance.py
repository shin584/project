"""Pinned real-checkpoint export and read-only e_system acceptance (#98).

Run from xai_test with --source, --consumer and a fresh --destination.
Missing acceptance inputs fail; synthetic substitutes and skips are not accepted.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import keras
import numpy as np

from SpCas9.spcas9_consumer_check import SOURCE_FILES, _consumer
from SpCas9.spcas9_export import ATOL, HEADS, RTOL, WINDOWS, export_checkpoint

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "xai_test/docs/evidence/issue97-consumer"
SOURCE_SHA256 = "24e31cc6b9df8520e2bf761d33f7bb475c4658a9a479242b67db882946d0d2b8"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path):
    stat = path.stat()
    return {
        "sha256": _sha256(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _consumer_parity(source, model, directory):
    # This observer forwards the exact array and return value, without casting,
    # fabricating outputs or replacing the Keras instance passed to run_prediction.
    calls = []
    real_predict = model.predict

    def observe(inputs, **kwargs):
        outputs = real_predict(inputs, **kwargs)
        assert inputs.dtype.kind == "i", "Consumer encoder must reach Keras uncast"
        assert inputs.shape == (1, 30, 4)
        assert type(outputs) is list and len(outputs) == 9
        for output in outputs:
            assert output.dtype == np.float32 and output.shape == (1, 1)
            assert np.isfinite(output).all()
        calls.append((inputs.copy(), outputs))
        return outputs

    encoded = np.array(
        [[[int(base == channel) for channel in "ACGT"] for base in w] for w in WINDOWS],
        dtype=np.float32,
    )
    named = source.predict(encoded, verbose=0)
    assert isinstance(named, dict), "Acceptance source must retain named outputs"
    with _consumer(directory) as (predict, scanner):
        sites = [
            scanner.CandidateSite(
                cas_type="SpCas9",
                strand="+",
                pam_start=24,
                cut_pos=21,
                distance=distance,
                guide_seq=window[4:24],
                pam_seq=window[24:26],
                model_input_seq=window,
                encoded_seq=["not the prediction encoder"],
            )
            for window, distance in zip(WINDOWS, (0, 10, 20), strict=True)
        ]
        result = scanner.ScanResult(input_seq="A" * 81, sites=sites)
        model.predict = observe
        try:
            assert predict.run_prediction(result, model, None, None) is result
        finally:
            model.predict = real_predict
    assert len(calls) == len(WINDOWS)
    assert len(result.variant_results) == len(WINDOWS) * 9
    cases = []
    for i, (site, (inputs, outputs)) in enumerate(zip(sites, calls, strict=True)):
        np.testing.assert_array_equal(inputs, encoded[i : i + 1])
        penalty = math.exp(-(site.distance**2) / 200)
        heads = []
        rows = result.variant_results[i * 9 : (i + 1) * 9]
        for j, ((protein, layer), row) in enumerate(zip(HEADS, rows, strict=True)):
            raw = float(outputs[j][0, 0])
            reference = float(named[layer][i, 0])
            assert row.cas_name == protein
            np.testing.assert_allclose(raw, reference, rtol=RTOL, atol=ATOL)
            assert row.raw_score == raw
            expected_final = raw * penalty * 100
            assert math.isclose(row.final_score, expected_final, rel_tol=1e-12)
            heads.append(
                {
                    "index": j,
                    "protein": protein,
                    "layer": layer,
                    "named_source": reference,
                    "raw": raw,
                    "source_absolute_error": abs(raw - reference),
                    "penalized_raw": raw * penalty,
                    "percentage_scale": 100,
                    "final": row.final_score,
                    "expected_final": expected_final,
                }
            )
        winner = max(rows, key=lambda row: row.final_score)
        assert site.raw_score == winner.raw_score
        assert site.final_score == winner.final_score
        cases.append(
            {
                "window": WINDOWS[i],
                "window_sha256": hashlib.sha256(WINDOWS[i].encode()).hexdigest(),
                "input": {
                    "dtype": str(inputs.dtype),
                    "dtype_kind": inputs.dtype.kind,
                    "shape": list(inputs.shape),
                    "encoded_array": inputs.tolist(),
                },
                "distance": site.distance,
                "penalty": penalty,
                "heads": heads,
                "candidate": asdict(site),
            }
        )
    return {
        "mode": "actual imported run_prediction with reloaded Keras model argument",
        "observation": "transparent predict observer; exact integer array forwarded uncast",
        "model_input_dtype": str(model.inputs[0].dtype),
        "keras_integer_input_accepted": True,
        "rtol": RTOL,
        "atol": ATOL,
        "cases": cases,
    }


def verify_acceptance(source, consumer, destination):
    """Export the agreed checkpoint and verify the saved artifact in e_system.

    report.json denotes completed acceptance only alongside the matching export
    artifact and manifest. Every attempt requires a fresh destination.
    """
    source = Path(source).resolve(strict=True)
    consumer = Path(consumer).resolve(strict=True)
    destination = Path(destination).resolve()
    initial = _identity(source)
    if initial["sha256"] != SOURCE_SHA256:
        raise ValueError("Acceptance source SHA-256 differs from the agreed manual run")
    if destination.is_relative_to(consumer) or destination.is_relative_to(
        source.parent
    ):
        raise ValueError(
            "Acceptance destination must be outside protected input directories"
        )
    prior = json.loads((EVIDENCE / "report.json").read_text(encoding="utf-8"))
    current_hashes = {name: _sha256(consumer / name) for name in SOURCE_FILES}
    selected = (
        consumer if current_hashes == prior["source_sha256"] else EVIDENCE / "source"
    )
    selected_hashes = {name: _sha256(selected / name) for name in SOURCE_FILES}
    assert selected_hashes == prior["source_sha256"], (
        "Preserved T2 source identity differs"
    )
    # Snapshot current bytes rather than requiring unrelated worktree edits to
    # match a historical hash. No datasets or evaluation reports are executed.
    prior_validation = json.loads(
        (EVIDENCE / "commit-validation.json").read_text(encoding="utf-8")
    )
    protected = {ROOT / name for name in prior_validation["preservation"]["files"]}
    protected.update(path for path in EVIDENCE.rglob("*") if path.is_file())
    protected.update(consumer / name for name in SOURCE_FILES)
    protected.add(source)
    before = {str(path): _sha256(path) for path in sorted(protected)}
    destination.mkdir()
    manifest = export_checkpoint(source, destination / "export")
    manifest_hash = _sha256(destination / "export/manifest.json")
    original = keras.models.load_model(source, compile=False)
    reloaded = keras.models.load_model(
        destination / "export/model.keras", compile=False
    )
    observations = _consumer_parity(original, reloaded, selected)
    after = {name: _sha256(name) for name in before}
    assert before == after, (
        "Protected source, split, log, evaluation or consumer bytes changed"
    )
    final = _identity(source)
    assert initial == final, "Source identity changed during acceptance"
    assert _sha256(destination / "export/model.keras") == manifest["artifact"]["sha256"]
    assert _sha256(destination / "export/manifest.json") == manifest_hash
    report = {
        "issue": 98,
        "status": "verified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "executable": sys.executable,
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{
                name: version(name)
                for name in ("numpy", "tensorflow", "keras", "torch", "streamlit")
            },
        },
        "artifact_directory": str(destination / "export"),
        "export": manifest,
        "export_manifest_sha256": manifest_hash,
        "source_final": final,
        "consumer_sources": {
            "current_directory": str(consumer),
            "current_sha256": current_hashes,
            "selected_directory": str(selected),
            "selected_sha256": selected_hashes,
            "used_preserved_t2": selected != consumer,
            "t2_report_sha256": _sha256(EVIDENCE / "report.json"),
            "t2_selection_status": prior["selection_status"],
        },
        "consumer": observations,
        "preservation": {
            "status": "unchanged",
            "file_count": len(before),
            "files": before,
        },
        "verification_code": {
            "module": "SpCas9.spcas9_export_acceptance",
            "sha256": _sha256(__file__),
            "consumer_import_helper_sha256": _sha256(
                Path(__file__).with_name("spcas9_consumer_check.py")
            ),
        },
        "follow_up": "https://github.com/shin584/project/issues/95",
        "limits": [
            "Format parity is not new training, improved accuracy, biological validation or deployment approval.",
            "No Testset evaluation, optimizer update, operating model selection or cache change executed.",
            "T2 all-nine final <= -1 crash/stale-state, nonfinite and 0-100 filter limits remain separate.",
            "The hardcoded cached v1 loader remains active; #95 repair, selection verification and return-to-v1 preparation precede adoption.",
        ],
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--consumer", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        verify_acceptance(args.source, args.consumer, args.destination)
    except (OSError, ValueError, AssertionError, RuntimeError) as exc:
        print(f"Acceptance unavailable or failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified real-checkpoint acceptance: {args.destination / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
