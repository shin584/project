# #179: disposable offline packaging gate

For the **production desktop / gate 2 (#188)**, use
[`docs/gate2/README.md`](../docs/gate2/README.md) and
`python packaging/build_desktop.py`. The final folder is `dist/XaiDashboard`;
the `XaiGate1` harness below remains a separate gate-1 experiment.

This is a packaging experiment, independent of the production desktop API (#178).
The recorded outcome is in [gate1-result.md](gate1-result.md).
The Streamlit screen remains the fallback. The predictor, model files, data,
desktop API and spike now all live in `xai_dashboard_desktop`. No sibling
checkout is required. The integrated predictor includes the offline loader
from `0ec27e1`, the #151 cache changes and the #180 cancellation callbacks.
See [integration details](../docs/integration-179-178-151-180.md).

## Decision: bundle a fixed WebView2 runtime

Loan-PC specifications are unavailable, so use the ticket's default: **bundle**
the x64 Fixed Version runtime. Do not rely on a system Evergreen installation or
an installer that downloads at first launch. This increases the copied folder's
size; measure that exact folder. The build still requires Windows' .NET Framework
4.8 for pywebview/pythonnet (present on supported Windows 10/11 installations).

Sources: [Microsoft distribution instructions](https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution),
[pywebview API](https://pywebview.flowrl.com/api/) (`WEBVIEW2_RUNTIME_PATH`,
`create_window(html=...)`, `http_server=False`),
[PyInstaller hooks](https://pyinstaller.org/en/stable/hooks.html).

## Build on Windows x64

Use Python 3.13 and a dedicated build environment. From `xai_dashboard_desktop`:

```powershell
python -m venv .packaging/venv
.packaging/venv/Scripts/python -m pip install -r packaging/requirements.txt
# Download the Microsoft Fixed Version x64 CAB and extract with expand.exe,
# preserving its versioned directory. Do not use Explorer's CAB extractor.
New-Item -ItemType Directory -Force .packaging/webview2
expand.exe <downloaded-runtime.cab> -F:* .packaging/webview2
.packaging/venv/Scripts/python packaging/prepare.py --webview2 <directory-containing-msedgewebview2.exe>
# Required build inputs: models/{Cas12a,SpCas9}.safetensors
.packaging/venv/Scripts/python -m PyInstaller --noconfirm packaging/gate1.spec
```

The preparation step may use the network; the executable cannot. It exports a
self-contained tokenizer, reloads it locally, and copies the complete fixed
runtime to `.packaging/assets`. Existing staging folders are rejected to avoid
mixing versions. Never commit weights, runtime files, or generated executables.

Copy **all** of `dist/XaiGate1`, including `_internal`, to the target PC.
Resource paths are relative to the bundle, never the working directory.
The runtime hook sets `HF_HUB_OFFLINE=1` before Hugging Face imports. The packaged
predictor loads only `_internal/tokenizer` with `local_files_only=True`.
Missing packaged weights raise immediately before any gdown import/download.
Development-mode downloads remain available; resource caching uses the
process-level cache from #151 without importing Streamlit in the predictor.

## Checks and evidence

```powershell
dist/XaiGate1/XaiGate1.exe --auto-close --output .packaging/developer-result.json
# Headless is diagnostic only and cannot pass the GUI check:
dist/XaiGate1/XaiGate1.exe --headless --output .packaging/headless-result.json
```

Each invocation creates an empty HF cache and blocks Python socket connects,
DNS resolution, and binds. This guard does **not** block native WebView2 network
access: a developer run is only a smoke test. Gate status stays
`pending_clean_environment` until the clean offline run below.

The fixed 81bp input contains a Cas12a TTTA and a SpCas9 AGG site. The harness
requests Cas12a first, then SpCas9(WT) through the lazy predictor. It verifies a
SHAP disk-cache miss (8 samples, 1 background) and observes actual inference
callbacks; runs IG (2 steps, chunk 1) and Rollout; checks finite arrays and
lengths; renders a matplotlib PNG data URI and confirms its decoded image width
in WebView. These low XAI settings test packaging, not explanation quality.

The result JSON records WebView2 aggregate RSS peak, application-tree RSS peak,
and TCP listeners sampled every 100 ms across the tree. Python bind attempts are
blocked as well. Monitor permission errors fail the smoke check. Memory is a
sampled aggregate RSS peak, not a private-working-set or historical OS peak.

## Clean, network-disabled validation

Use Windows Sandbox (if available) to avoid changing the developer's network,
user account, caches, or checkpoints:

```powershell
powershell -NoProfile -File packaging/new-sandbox.ps1
# Open .packaging/sandbox/gate1.wsb
```

The generated configuration disables networking and maps only the bundle,
validation scripts, and an empty results folder. The validator checks that
Python/HF cache/external models are absent, copies the bundle to the Sandbox
desktop, and runs it from an unrelated working directory. It writes
`gate1-result.json` and `environment.json` to `.packaging/sandbox/results` on the
host. Only this validator can promote a smoke pass to a gate pass.

If Sandbox is unavailable, use a fresh Windows VM/account with Python absent,
no HF cache or external models, networking disabled before launching the copied
folder, and record these conditions alongside the result JSON. Do not mark the
gate passed based on a developer PC or a headless run. Failures retain the
Streamlit fallback; #151 and #154 continue independently.

Run the integrated core, desktop API and Streamlit regression tests:

```powershell
python -X utf8 -m pytest test tests -q
```
