import hashlib
import json
import subprocess
import sys
import zipfile

import pytest

from xai_test.cas12a_NT_analysis.nt_colab import build_bundle
from xai_test.cas12a_NT_analysis.nt_validation import audit_evidence


def test_handoff_contains_exact_source_and_compilable_guarded_notebook(tmp_path):
    root = tmp_path / "bundle"
    result = build_bundle(root)
    assert set(result["files"]) == {
        "cas12a-validation-source.zip",
        "Cas12a_NT_validation_colab.ipynb",
    }
    notebook = json.loads((root / "Cas12a_NT_validation_colab.ipynb").read_bytes())
    source_sha = result["files"]["cas12a-validation-source.zip"]
    sources = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    for index, source in enumerate(sources):
        compile(source, f"colab-cell-{index}", "exec")
    joined = "\n".join(sources)
    assert source_sha in joined
    assert 'APPROVED_PROTOCOL_SHA256 = ""' in joined
    assert "FIRST_EVER_STUDY = False" in joined
    assert "SINGLE_EXECUTOR = False" in joined
    assert "RECOVER_INTERRUPTED_JOB = False" in joined
    assert "git clone" not in joined
    with zipfile.ZipFile(root / "cas12a-validation-source.zip") as archive:
        manifest = json.loads(archive.read("source-manifest.json"))
        for name, checksum in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum
        assert "xai_test/cas12a_NT_analysis/nt_validation.py" in manifest["files"]
        assert "cas12a/nt_data.py" in manifest["files"]
        assert not any(
            "model.safetensors" in name or name.endswith(".xlsx")
            for name in archive.namelist()
        )
    assert build_bundle(tmp_path / "another") == result
    extracted = tmp_path / "standalone"
    with zipfile.ZipFile(root / "cas12a-validation-source.zip") as archive:
        archive.extractall(extracted)
    code = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from xai_test.cas12a_NT_analysis import nt_colab
assert Path(nt_colab.__file__).resolve().is_relative_to(Path(sys.argv[1]))
print(json.dumps(nt_colab.build_bundle(sys.argv[2])))
"""
    portable = subprocess.run(
        [sys.executable, "-I", "-c", code, str(extracted), str(tmp_path / "portable")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(portable.stdout) == result
    with pytest.raises(ValueError, match="new empty"):
        build_bundle(root)


def test_returned_archive_rejects_corruption_and_path_escape(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        data = b"escape"
        archive.writestr("../escape", data)
        archive.writestr(
            "manifest.json",
            json.dumps({"files": {"../escape": hashlib.sha256(data).hexdigest()}}),
        )
    checksum = hashlib.sha256(bad.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="escapes|unsafe"):
        audit_evidence(bad, checksum=checksum, directory=tmp_path / "audit")
    with pytest.raises(ValueError, match="SHA256"):
        audit_evidence(bad, checksum="0" * 64, directory=tmp_path / "audit")
    assert not (tmp_path / "escape").exists()
