# Build via packaging/build_desktop.py with staged and verified inputs.
# ruff: noqa: F821
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

root = Path(SPECPATH).parent
assets = Path(os.environ["DASHBOARD_V3_STAGED_ASSETS"])
datas = [
    (str(assets / name), name)
    for name in ("models", "tokenizer", "webview2")
]
datas += [
    (str(root / "docs/equivalence-baseline" / name), "equivalence-baseline")
    for name in ("capture.json", "candidates.json", "crispr_prediction_results.csv")
]
datas += copy_metadata("shap")
hiddenimports = (
    collect_submodules("transformers.models.esm")
    + collect_submodules("transformers.models.auto")
    + collect_submodules("shap")
    + collect_submodules("keras")
    + ["matplotlib.backends.backend_agg", "desktop.worker_renderer", "desktop.fake",
       "non_nt_predict", "non_nt_cas12a_predict"]
)
a = Analysis(
    [str(root / "desktop_entry.py")], pathex=[str(root)], datas=datas,
    hiddenimports=hiddenimports, runtime_hooks=[str(root / "packaging/runtime_hook.py")],
    excludes=["streamlit", "tensorflow", "tensorboard", "IPython", "jupyter",
              "pytest", "torchvision", "torchaudio", "PyQt5", "PyQt6", "PySide2",
              "PySide6", "tkinter", "gdown"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="XaiDashboardV3",
          console=False, debug=False, strip=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="XaiDashboardV3")
