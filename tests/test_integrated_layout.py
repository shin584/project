"""The merged checkout must work without sibling repositories or online fallbacks."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import model_source
import predict
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_core_and_fake_work_in_an_isolated_copy(tmp_path):
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    for name in ("pam_scanner.py", "predict.py", "model_source.py"):
        shutil.copy2(ROOT / name, standalone / name)
    for name in ("desktop", "xai"):
        shutil.copytree(
            ROOT / name, standalone / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    script = '''
import importlib.abc
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

class RejectModelAndUIImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"streamlit", "torch", "transformers", "safetensors"}:
            raise AssertionError("Unexpected dependency: " + fullname)

sys.meta_path.insert(0, RejectModelAndUIImports())
import pam_scanner
import predict
from desktop.fake import FakeRenderer
from xai import explain

assert Path(pam_scanner.__file__).parent == Path(sys.argv[1])
assert all(p._model is None for p in predict.load_predictors(warmup=False).values())
dna = "GATCCAGTCAGCATGC" + "TTTA" + "CGATCGTACGATCGAATCGT" + "AGG" + "CTAGCATCGATCGATGCATCGTAGCTAGCTAGCATCGA"
fragments = FakeRenderer(step_delay=0).prediction_fragments(dna, lambda d, t: None)
assert set(fragments) == {"best", "variants", "detail"}
assert all(fragments.values())
'''
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(standalone)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_packaged_tokenizer_stays_local_and_cached(monkeypatch, tmp_path):
    (tmp_path / "tokenizer").mkdir()
    calls = []
    tokenizer = object()

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return tokenizer

    monkeypatch.setattr(predict, "is_packaged", lambda: True)
    monkeypatch.setattr(predict, "resource_root", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained),
    ))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(name, "0")
    predict.load_nt_tokenizer.cache_clear()
    try:
        assert predict.load_nt_tokenizer() is tokenizer
        assert predict.load_nt_tokenizer() is tokenizer
        assert calls == [(str(tmp_path / "tokenizer"), {"local_files_only": True})]
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    finally:
        predict.load_nt_tokenizer.cache_clear()


def test_packaged_missing_model_fails_before_downloading(monkeypatch, tmp_path):
    monkeypatch.setattr(model_source, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(model_source, "is_packaged", lambda: True)

    def unexpected():
        pytest.fail("Packaged loading must not attempt a download")

    monkeypatch.setattr(model_source, "_require_gdown", unexpected)
    with pytest.raises(FileNotFoundError, match="Offline bundle is missing model"):
        model_source.resolve("models/SpCas9.safetensors")


def test_packaged_missing_tokenizer_never_falls_back(monkeypatch, tmp_path):
    (tmp_path / "tokenizer").mkdir()
    calls = []

    def missing(path, **kwargs):
        calls.append((path, kwargs))
        raise OSError("Missing exported tokenizer")

    monkeypatch.setattr(predict, "is_packaged", lambda: True)
    monkeypatch.setattr(predict, "resource_root", lambda: tmp_path)
    monkeypatch.setattr(predict, "configure_offline", lambda: None)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=missing),
    ))
    predict.load_nt_tokenizer.cache_clear()
    try:
        with pytest.raises(OSError, match="Missing exported tokenizer"):
            predict.load_nt_tokenizer()
        assert calls == [(str(tmp_path / "tokenizer"), {"local_files_only": True})]
    finally:
        predict.load_nt_tokenizer.cache_clear()
