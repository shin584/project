"""Core imports and resource caching must work with no UI or model files."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import predict


def test_core_imports_without_ui_framework_or_model_libraries():
    script = '''
import importlib.abc
import sys

class RejectModelAndUIImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"streamlit", "torch", "transformers", "safetensors"}:
            raise AssertionError("Unexpected dependency during core import: " + fullname)

sys.meta_path.insert(0, RejectModelAndUIImports())
import predict
from xai import explain

predictors = predict.load_predictors(warmup=False)
assert all(p._model is None for p in predictors.values())
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tokenizers_are_reused_by_repository_without_downloading(monkeypatch):
    downloads = []

    def from_pretrained(repo):
        downloads.append(repo)
        return SimpleNamespace(repo=repo)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained),
    ))
    predict.load_nt_tokenizer.cache_clear()
    try:
        first = predict.load_nt_tokenizer()
        for _ in range(5):
            assert predict.load_nt_tokenizer() is first
        other = predict.load_nt_tokenizer("another/repository")
        assert other is not first
        assert predict.load_nt_tokenizer("another/repository") is other
        assert downloads == [predict.NT_TOKENIZER_REPO, "another/repository"]
    finally:
        predict.load_nt_tokenizer.cache_clear()
