# python -m PyInstaller --noconfirm packaging/desktop.spec
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH).parent
assets = root / '.packaging' / 'assets'
required = [assets / 'tokenizer' / 'vocab.txt',
            assets / 'webview2' / 'msedgewebview2.exe',
            root / 'models' / 'Cas12a.safetensors',
            root / 'models' / 'SpCas9.safetensors',
            # Adopted non-NT checkpoints (#191). Keras 3 reads these under the
            # torch backend; see docs/runtime/README.md (#193).
            root / 'models' / 'non_nt' / 'spcas9-9hydra-v2' / 'model.keras',
            root / 'models' / 'non_nt' / 'cas12a-cnn-20261102' / 'cas12a_cnn_manifest.keras']
for path in required:
    if not path.is_file():
        raise SystemExit(f'Missing packaging input: {path}; see packaging/README.md')

datas = [(str(assets / 'tokenizer'), 'tokenizer'),
         (str(assets / 'webview2'), 'webview2'),
         (str(required[2]), 'models'), (str(required[3]), 'models'),
         (str(required[4]), 'models/non_nt/spcas9-9hydra-v2'),
         (str(required[5]), 'models/non_nt/cas12a-cnn-20261102'),
         (str(root / 'data' / 'precomputed_shap'), 'data/precomputed_shap'),
         (str(root / 'docs' / 'equivalence-baseline'), 'equivalence-baseline')]
# Keras 3 resolves layers and savers through dynamic imports, so the whole
# package ships rather than a hand-listed set (#193). TensorFlow stays excluded
# below: `keras.src.saving` only *probes* for it through a LazyModule, so the
# probe resolves to False in the bundle and the torch backend carries the
# forward pass and the IG gradients alone.
hiddenimports = (collect_submodules('transformers.models.esm') +
                 collect_submodules('transformers.models.auto') +
                 collect_submodules('shap') + collect_submodules('keras') +
                 ['matplotlib.backends.backend_agg',
                 'desktop.worker_renderer', 'desktop.fake', 'non_nt_runtime'])
a = Analysis([str(root / 'desktop_entry.py')], pathex=[str(root)],
             datas=datas, hiddenimports=hiddenimports,
             runtime_hooks=[str(root / 'packaging' / 'runtime_hook.py')],
             excludes=['streamlit', 'tensorflow', 'tensorboard', 'IPython',
                       'jupyter', 'pytest', 'torchvision', 'torchaudio', 'PyQt5',
                       'PyQt6', 'PySide2', 'PySide6', 'tkinter'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='XaiDashboard',
          console=False, debug=False, strip=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='XaiDashboard')
