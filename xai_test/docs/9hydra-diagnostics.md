# 9-Hydra v2 diagnostic execution

Issue #69 supplies the execution boundary for the verification in #60. Run
`SpCas9.spcas9_diagnostics.run_diagnostics(artifacts, destination, configuration)`
to receive a structured report and preserve its evidence in a fresh directory.
`ArtifactLocations` requires the dataset, published split directory, existing
checkpoint, historical training log, saved predictions and prediction sidecar.
Its optional `evidence` tuple accepts additional files or directories, including
comparison results and their sidecars. Every file within an input directory is
included in the before/after SHA-256 record.

The stage registry includes backend observation, artifact validation, checkpoint
parity, copy save/reload, mask/metric/gradient checks, aggregation and finding
reports. Only environment observation and harness checks are implemented here.
The remaining stages report `blocked` until subsequent tickets implement them.
A normal #69 run therefore returns `incomplete`; it does not verify the model.
This ticket does not run the full original-artifact diagnostic reserved for
ticket 10 of the approved breakdown.

From `xai_test/`, invoke the explicitly chosen interpreter, for example in PowerShell:

```powershell
& 'C:/Users/cmino/AppData/Local/Programs/Python/Python313/python.exe' -m SpCas9.spcas9_diagnostics `
  --dataset dataset/kim_2020_table_s8.xlsx `
  --split-dir SpCas9/published_split `
  --checkpoint SpCas9/v2_train/9hydra_v2.keras `
  --training-log SpCas9/v2_train/training_log.json `
  --predictions SpCas9/v2_eval/predictions.csv `
  --prediction-sidecar SpCas9/v2_eval/predictions_checksum.json `
  --destination diagnostic-runs/verification-001
```

The CLI exits with status 1 for `incomplete` or `failed`, and 0 only for
`verified`. A destination collision is rejected before any outputs are written.
Source directories, source files and aliases into source directories cannot
serve as output destinations. Existing outputs are never cleaned up or resumed.

`DiagnosticConfig` freezes the interpreter, requested device, inference batch
size (default 1024), fixture specification, command and relevant environment
settings before execution. Changes to those settings after freezing are rejected.
Tolerances are fixed at `rtol=1e-5, atol=1e-6`; neither constructor nor CLI accepts
overrides. The default fixture specification explicitly marks the numerical
fixture as deferred; later runtime tickets must freeze its exact finite values
and independent expected arithmetic before their probes run.

Each destination contains:

- `configuration.json`: the frozen execution configuration, including unset
  environment variables as `null` with `unset/default` status.
- `report.json`: checks with completion state, finding disposition, evidence,
  required/in-scope flags and timestamps, plus the overall execution outcome.
- `summary.txt`: the readable report, also printed to standard output.
- `manifest.json`: revision, dirty status, source snapshot before/after hashes,
  interpreter/Python, installed dependency versions, backend observations,
  requested device, commands, timestamps, input before/after hashes and output
  hashes. The manifest excludes its own hash. Installed versions are provenance,
  not evidence that a checkpoint executed successfully.

Completion states are `passed`, `failed`, `blocked` and `not applicable`.
Finding dispositions are recorded separately: `no defect established`,
`explained behavior`, `repaired defect`, `unresolved defect`,
`unresolved model limitation` and `excluded scope`. A required in-scope failure
makes the execution `failed`; any other required in-scope result without a pass
makes it `incomplete`. A model limitation can be a completed investigation
without being a repaired defect.

Independent results survive another stage's failure; a stage with a non-passing
prerequisite is blocked without execution. Missing backends or inputs remain
explicitly incomplete. Stage failures and interruptions retain partial files and
complete the input preservation record. Source mutation is reported as a failed
execution, with no automatic restoration or replacement.

The backend observer reports effective oneDNN and determinism as `unknown` when
it cannot observe them. Unset `TF_ENABLE_ONEDNN_OPTS` does not establish whether
oneDNN was enabled. Thread counts of zero from TensorFlow mean runtime defaults.
Raw environment settings remain available alongside these observations.

The diagnostic path exposes no training operation and guards synchronous stage
execution against `fit`, smoke-training CLI calls, optimizer updates and study
creation/execution. This guard catches accidental adapter calls; it is not a
sandbox for arbitrary code or background worker threads. The #51 study guard
and its candidate-selection restrictions remain in force. Boundary tests use
small byte fixtures and backend doubles for harness/error paths; these results
do not satisfy the later real-artifact or real-backend acceptance checks.
