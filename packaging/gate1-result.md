# #179 gate 1 implementation and validation

This is the original #179 validation record, before the #151/#180 integration.
For the current local layout and subsequent validation, see
[the integration record](../docs/integration-179-178-151-180.md).

Date: 2026-11-16. Shared predictor commit: `0ec27e1` (`dashboard_core`).
All new source files were under `xai_dashboard_desktop` on `model_analysis`
at the time (`24506c1`). They now live at the repository root of
`dashboard_core_v2` (`51fd7a1`); see the integration record for the mapping.

## Validation completed

- Shared dashboard subtree: **45 passed**, one existing KernelSHAP conditioning
  warning. Command: `PYTHONUTF8=1 python -m pytest ../xai_dashboard/tests -q`.
- Real exported ESM tokenizer loaded with a new empty HF cache and Python
  socket connects/DNS/binds blocked. Missing tokenizer fails without fallback.
- Missing packaged SpCas9 weights raise `FileNotFoundError` with no gdown import
  or network request. Existing development loading behavior is preserved.
- New Python files pass Ruff and compile checks. Both PowerShell scripts parse.
- Sandbox configuration generation was exercised; its XML sets networking to
  `Disable`. A copied executable with its tokenizer vocabulary temporarily
  removed failed immediately, before loading models or WebView2; it was restored.

## Build and executable evidence

**One-folder build and copied executable smoke: PASS.** The final build was
checked to contain the committed loader's tokenizer initialization order.
The complete folder was copied to `.packaging/copied/XaiGate1` and run from an
empty working directory with only Windows directories on PATH and a fresh HF
cache. The executable exited with code 0.

| Check | Final observed result |
|---|---|
| Cas12a prediction | 1 result |
| SpCas9 lazy prediction, AGG site | 8 variant results, including SpCas9(WT) |
| Uncached KernelSHAP | 8 samples / 1 background, 3 inference callbacks |
| IG backward | 2 steps, chunk size 1 |
| Attention Rollout | 24 layers |
| Matplotlib PNG in WebView | Image decoded and displayed |
| Process-tree TCP listeners | 0 |
| Blocked Python network attempts / monitor errors | 0 / 0 |
| Sampled WebView2 aggregate RSS peak | 305,274,880 bytes (291.13 MiB) |
| Sampled application-tree aggregate RSS peak | 2,949,451,776 bytes (2.75 GiB) |
| Harness elapsed time | 112.73 seconds (one observation, not a performance benchmark) |
| Complete distribution | 6,911 files, 5,498,628,576 bytes (5.12 GiB) |

Raw final results: [developer-smoke.json](developer-smoke.json).
Build identity: [build-manifest.json](build-manifest.json).
Memory uses 100 ms RSS samples, not private bytes or historical OS peaks.
The low SHAP/IG settings test execution and packaging, not explanation quality.
The existing singular-regression SHAP warning was emitted; results were finite.

Build environment: Windows 11 Pro x64, 7.8 GiB RAM, Python 3.13.7,
PyInstaller 6.22.3 / hooks 2026.10, torch 2.12.0, transformers 5.8.1,
SHAP 0.52.0, numba 0.68.0 / llvmlite 0.50.0, numpy 2.3.2,
Streamlit 1.51.0, matplotlib 3.10.7, psutil 7.2.2.
Fixed WebView2 **157.0.4341.60 x64** is bundled; no Evergreen fallback is used.

## Clean offline gate

**Not passed: clean-environment validation is unavailable on this host.**
`C:\Windows\System32\WindowsSandbox.exe` is absent. The developer machine has
Python, original model files, and an HF cache. An empty process-local cache and
Python socket guard do not satisfy the clean-PC/network-disabled requirement:
native WebView2 is outside that guard.

Use `packaging/new-sandbox.ps1` on a Sandbox-capable Windows host, or a fresh
offline Windows VM. Preserve `environment.json` and `gate1-result.json` from the
copied-folder run. Until that run passes, retain the Streamlit demonstration
fallback; #151 and #154 are independent. Do not close #179 as a passed gate.

Generated build logs and raw measurements stay under the ignored
`.packaging/evidence` directory; the executable folder stays under `dist/XaiGate1`.

## Standards

Independent reviewer: no documented-standard violations. New files stay under
`xai_dashboard_desktop`; the Streamlit screen and its cache decorators remain.
One low-priority judgement call: the monitor failure condition is repeated in
`run()` and `main()`. Sharing that condition would be optional cleanup for this
disposable spike.

## Spec

Independent reviewer: no confirmed implementation defects. The spike covers
the fixed-runtime policy, bundled offline tokenizer, fail-fast model loading,
both lazy model paths, uncached SHAP, IG, Rollout, data-URI rendering, and
memory/port monitoring. Actual clean offline execution remains an acceptance
gap; a developer smoke pass cannot replace it. The parent-issue result must
record this limitation.

Review totals: Standards 0 hard violations / 1 optional smell; Spec 0 confirmed
code defects / clean-environment acceptance still pending.
