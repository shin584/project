"""The `core` facade names the core's public surface without moving any file."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import core

ROOT = Path(__file__).resolve().parent.parent


def _run_python(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT,
        capture_output=True, text=True, timeout=60, check=False,
    )


def test_import_core_pulls_no_ui_framework_or_model_library():
    result = _run_python('''
import importlib.abc
import sys

class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"streamlit", "torch", "transformers", "safetensors"}:
            raise AssertionError("Unexpected dependency during core import: " + fullname)

sys.meta_path.insert(0, Reject())
import core
for name in core.__all__:
    getattr(core, name)
''')
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_core_defers_chart_libraries_until_a_chart_is_used():
    result = _run_python('''
import sys
import core
assert "shap" not in sys.modules and "matplotlib" not in sys.modules, "eager chart import"
core.shap_bars
assert "shap" in sys.modules
''')
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_public_name_is_the_object_from_its_unmoved_module():
    for name in core.__all__:
        module_name, attr = core._EXPORTS[name]
        source = importlib.import_module(module_name)
        assert source.__file__ is not None
        assert not Path(source.__file__).resolve().is_relative_to(ROOT / "core"), module_name
        assert getattr(core, name) is getattr(source, attr), name


def test_type_checking_imports_mirror_the_runtime_exports():
    tree = ast.parse(Path(core.__file__).read_text(encoding="utf-8"))
    guard = next(n for n in tree.body if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == "TYPE_CHECKING")
    static = {
        (alias.asname or alias.name): (node.module, alias.name)
        for node in guard.body if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert static == core._EXPORTS


def test_public_surface_covers_scan_predict_explain_and_charts():
    expected = {
        "scan_all", "validate_sequence", "CandidateSite", "VariantResult", "ScanResult",
        "load_predictors", "run_prediction", "MODEL_REGISTRY",
        "integrated_gradients", "kernel_shap", "attention_rollout",
        "shap_cache_key", "load_cached_shap", "save_cached_shap",
        "attribution_logo", "shap_bars", "shap_waterfall",
    }
    assert expected <= set(core.__all__)
    assert set(core.__all__) == set(core._EXPORTS)
    assert set(core.__all__) <= set(dir(core))


def test_private_or_unknown_names_are_not_reachable_through_core():
    import predict
    assert hasattr(predict, "_to_display_score")  # 실제로 있는 내부 이름이어야 의미가 있다
    with pytest.raises(AttributeError):
        core._to_display_score
    with pytest.raises(AttributeError):
        core.no_such_name
