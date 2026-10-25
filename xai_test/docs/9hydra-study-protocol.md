# 9-Hydra study protocol

`SpCas9.spcas9_study.StudyProtocol` is the experiment boundary introduced by
[#51](https://github.com/shin584/project/issues/51). It extends the existing v2
training, named-head prediction and measured-row report code. Historical
commands and artifacts retain their behavior.

Planning verifies the source spreadsheet, the exact checksum-sidecar bytes and
the CSV bytes against independent pins. It requires 29,134 train, 3,676 validation
and 5,186 Test rows. A mismatch is an error; this API never regenerates a split.
`StudyInputs` and `InputPins` allow explicitly pinned fixtures or relocated
inputs. Default pins identify the existing published artifacts. Checking Test
integrity does not permit Test prediction or tuning.

From `xai_test/`, planning and inspecting a run require no training framework:

```python
from pathlib import Path
from SpCas9.spcas9_study import StudyProtocol

study = StudyProtocol.plan(Path("SpCas9/studies"))
run_id = study.plan_run(seed=42)
planned = study.report_only(run_id)
```

After real execution is authorized by the study execution ticket (#59), use
`study.train_validation(run_id)` for the Keras backend. Fixture tests inject a
`TrainingBackend` via `backend=` instead. Planning does not start a fit. Only
v2 is implemented here; R2/R3, bounded scheduling, selection freeze, final Test
evaluation and the complete study handoff are subsequent tickets. The exposed
`freeze_selection()` raises `NotImplementedError`; `evaluate_test()` raises
`PermissionError` until the finalist-freeze stage is implemented. There is no
Test prediction path in this ticket.

The backend receives only encoded sequences and named labels/weights. The
architecture, linear heads, percentage/100 labels, binary measured-label masks,
equal head coefficients and original optimizer settings are preserved. tRNA and
PAM-library membership are analysis metadata. No labels or predictions are
clipped, and repeated sequences remain separate split rows.

Checkpoint selection and LR reduction both monitor
`val_reference_masked_mse`: the sum of squared errors over measured slots of all
nine heads, divided by the validation row count. Missing slots never contribute.
`recipe_optimization_loss` and `val_recipe_optimization_loss` are separately
named in history. Early stopping restores the best reference-loss weights;
the protocol checks that checkpoint predictions reproduce this loss. Save/reload
parity uses all validation sequences and never Test sequences.

Each study and run gets a fresh location; supplying an existing identifier is
refused. Run states distinguish `planned`, `running`, `completed` and `failed`.
The protocol reserves a run exclusively and never retries in the same output
directory. Failed/interrupted runs keep their partial artifacts and error
metadata with `eligible: false`. A process killed without exception handling
may remain `running`, which is also ineligible. Restart recovery and attempt/time
budgets belong to #55. `eligible: true` means complete, verified run evidence;
it is not a finalist selection or an adoption decision.

The study stores `protocol.json` and `study.json` with the protocol hash,
revision, dirty state, source-file digest inventory and snapshot identity,
Python/platform and installed library versions. Each run contains:

- `plan.json`, `started.json`, `execution.json` and `state.json`: seed, lifecycle
  and actual execution provenance, including consumed input digests.
- `9hydra_v2.keras` and `training.json`: restored best checkpoint, stopping
  details, separately named history, environment/device, elapsed time and parity.
- `predictions.csv`: validation split-row identity, sequence and analysis
  metadata, with labels and predictions named by head. Missing labels stay missing.
- `validation.json` and `evidence.json`: measured-row/stratified metrics and
  hashes of all completed run artifacts, including checkpoint and predictions.

`StudyProtocol(study.path).report_only(run_id)` verifies the persisted evidence
and rebuilds validation metrics from saved predictions without importing the
training backend or loading a model. Corrupted artifacts raise an error; they
cannot be returned as an eligible completed run. Report-only needs no source
dataset access. Keep the whole output directory: CSVs and checkpoints may be
ignored by Git, so a code commit alone does not retain run evidence.
