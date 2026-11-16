# python -m PyInstaller --noconfirm packaging/desktop.spec
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH).parent
assets = root / '.packaging' / 'assets'
required = [assets / 'tokenizer' / 'vocab.txt',
            assets / 'webview2' / 'msedgewebview2.exe',
            root / 'models' / 'Cas12a.safetensors',
            root / 'models' / 'SpCas9.safetensors']
for path in required:
    if not path.is_file():
        raise SystemExit(f'Missing packaging input: {path}; see packaging/README.md')

datas = [(str(assets / 'tokenizer'), 'tokenizer'),
         (str(assets / 'webview2'), 'webview2'),
         (str(required[2]), 'models'), (str(required[3]), 'models'),
         (str(root / 'data' / 'precomputed_shap'), 'data/precomputed_shap'),
         (str(root / 'docs' / 'equivalence-baseline'), 'equivalence-baseline')]
hiddenimports = (collect_submodules('transformers.models.esm') +
                 collect_submodules('transformers.models.auto') +
                 collect_submodules('shap') + ['matplotlib.backends.backend_agg'])
a = Analysis([str(root / 'desktop_entry.py')], pathex=[str(root)],
             datas=datas, hiddenimports=hiddenimports,
             runtime_hooks=[str(root / 'packaging' / 'runtime_hook.py')],
             excludes=['streamlit', 'tensorflow', 'keras', 'tensorboard', 'IPython',
                       'jupyter', 'pytest', 'torchvision', 'torchaudio', 'PyQt5',
                       'PyQt6', 'PySide2', 'PySide6', 'tkinter'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='XaiDashboard',
          console=False, debug=False, strip=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='XaiDashboard')
