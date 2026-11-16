# Build from xai_dashboard_desktop: python -m PyInstaller packaging/gate1.spec
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

root = Path(SPECPATH).parent
core = root
assets = root / '.packaging' / 'assets'
required = [assets / 'tokenizer' / 'vocab.txt',
            assets / 'webview2' / 'msedgewebview2.exe',
            core / 'models' / 'Cas12a.safetensors',
            core / 'models' / 'SpCas9.safetensors']
for path in required:
    if not path.is_file():
        raise SystemExit(f'Missing packaging input: {path}; see packaging/README.md')

datas = [(str(assets / 'tokenizer'), 'tokenizer'),
         (str(assets / 'webview2'), 'webview2'),
         (str(required[2]), 'models'), (str(required[3]), 'models')]
# AutoTokenizer and ESM use lazy imports. Collect their concrete implementations.
hiddenimports = (collect_submodules('transformers.models.esm') +
                 collect_submodules('transformers.models.auto') +
                 collect_submodules('shap') + ['matplotlib.backends.backend_agg'])
datas += copy_metadata('streamlit')
a = Analysis([str(root / 'packaging_spike.py')], pathex=[str(root)],
             datas=datas, hiddenimports=hiddenimports,
             runtime_hooks=[str(root / 'packaging' / 'runtime_hook.py')],
             excludes=['tensorflow', 'keras', 'tensorboard', 'IPython', 'jupyter',
                       'pytest', 'torchvision', 'torchaudio', 'PyQt5', 'PyQt6',
                       'PySide2', 'PySide6', 'tkinter'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='XaiGate1',
          console=True, debug=False, strip=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='XaiGate1')
