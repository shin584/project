"""Preparation/recovery failures at the supported operator CLI."""

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def cli(script, *args):
    return subprocess.run([sys.executable, str(ROOT / "docs/runtime" / script), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


@pytest.fixture
def bundle(tmp_path):
    work = tmp_path / "bundle"
    result = cli("verify_contract.py", "init", "--work", work, "--run-id", "recovery-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    return work


def test_preparation_requires_declared_wheels_instead_of_personal_packages(bundle, tmp_path):
    wheels = tmp_path / "empty-wheels"
    wheels.mkdir()
    result = cli("stage_offline.py", "prepare", "--work", bundle, "--wheelhouse", wheels,
                 "--output", tmp_path / "prepared")
    assert result.returncode == 1
    assert "Missing/mismatched declared wheel: keras==3.14.1" in result.stdout
    assert not (tmp_path / "prepared/preparation.json").exists()


def test_recovery_requires_explicit_preparation(bundle, tmp_path):
    result = cli("stage_offline.py", tmp_path / "restored", "--work", bundle,
                 "--output", bundle / "staged.json")
    assert result.returncode != 0
    assert "--prepared" in result.stderr
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("fault", ["mixed", "checksum", "source", "overlap"])
def test_bad_preparation_is_recorded_as_failed_offline_stage(bundle, tmp_path, fault):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    checksum = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    files = {}
    for name, item in manifest["sources"].items():
        relative = ("payload/" + name.split("/", 1)[1] if name.startswith(("code/", "assets/"))
                    else "payload/recovery/" + name)
        files[relative] = item["sha256"]
    files["payload/recovery/inputs.npz"] = manifest["inputs_sha256"]
    data = {"bundle_manifest_sha256": checksum, "files": files, "original_roots": [str(ROOT)],
            "wheels": {}, "original_files": []}
    restored = tmp_path / "restored"
    expected = "Prepared input checksum mismatch"
    if fault == "mixed":
        data["bundle_manifest_sha256"] = "0" * 64
        expected = "different bundle"
    elif fault == "source":
        data["files"]["payload/non_nt_runtime.py"] = "0" * 64
        expected = "disagrees with frozen bundle"
    elif fault == "overlap":
        restored = ROOT / "refuse-recovery-fixture"
        expected = "independent"
    else:
        # Include one corrupt payload so validation fails before installation.
        for relative in files:
            target = prepared / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"corrupt")
    (prepared / "preparation.json").write_text(json.dumps(data), encoding="utf-8")
    result = cli("stage_offline.py", restored, "--prepared", prepared,
                 "--work", bundle, "--output", bundle / "staged.json")
    assert result.returncode == 1
    document = json.loads((bundle / "staged.json").read_text(encoding="utf-8"))
    assert document["run"]["status"] == "fail"
    assert expected in document["run"]["error"]
    receipt = json.loads((bundle / "offline.receipt.json").read_text(encoding="utf-8"))
    assert receipt["exit_code"] == 1
    assert not restored.exists()


def test_transported_controller_can_record_failure_without_original_sources(bundle, tmp_path):
    """A missing preparation fails cleanly even after every original path is gone."""
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, item in enumerate(manifest["sources"].values()):
        item["source"] = str(tmp_path / "unavailable-original" / str(index))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "manifest.sha256").write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest(), encoding="ascii")
    controller = bundle / "snapshot/code/docs/runtime/stage_offline.py"
    result = subprocess.run([sys.executable, "-S", str(controller), str(tmp_path / "restored"),
        "--prepared", str(tmp_path / "missing-prepared"), "--work", str(bundle),
        "--output", str(bundle / "staged.json"), "--snapshot-only"],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads((bundle / "offline.receipt.json").read_text(encoding="utf-8"))
    assert receipt["input_mode"] == "snapshot"
    assert receipt["exit_code"] == 1
    assert json.loads((bundle / "journal.json").read_text(encoding="utf-8"))["state"] == "ready"
