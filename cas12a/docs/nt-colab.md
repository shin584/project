# NT-Cas12a Colab execution handoff

[#109](https://github.com/shin584/project/issues/109) supplies
[NT_Cas12a_colab.ipynb](../model/NT_Cas12a_colab.ipynb), a thin caller of
`cas12a.nt_workflow.NTWorkflow`. **Notebook preparation and offline checks are
implementation readiness, not actual NT-500M training.** Production GPU,
full-library inference, Drive authorization/copy and final measurements remain
unexecuted until a human session produces their evidence. This handoff does
not launch an actual run, purchase GPU capacity or provision Windows CUDA.

## Human execution boundaries

1. Publish the reviewed implementation commit or upload its checkout. Open the
   notebook in Colab, select a GPU runtime and connect. Set `CODE_REF` to that
   commit; keep `REPO_ROOT` on the VM under `/content`. For an uploaded checkout,
   skip the clone cell and set the working directory/import path to that copy;
   record its commit and any changes. Never replace code during an active run.
2. Execute environment installation, then restart the Python session when
   required before importing the workflow. Re-run source setup, skip installation
   and authorize `drive.mount` interactively. Do not use Run all before configuring
   the run and inspecting each stage.
3. Supply the original source and locations, choose fresh/resume explicitly,
   inspect preparation/preflight evidence, and execute the training cell when
   ready. Only this human action starts production fitting. Inspect each later
   stage before advancing. Do not automatically run another experiment after
   final evaluation.

## Fixed environment and configurable locations

[requirements-nt.txt](../requirements-nt.txt) pins the direct runtime dependency
set verified by real CPU fixture runs: torch 2.12.0, Transformers 5.8.1,
tokenizers 0.22.2, huggingface-hub 1.15.0, safetensors 0.7.0, NumPy 2.3.2,
pandas 2.3.3, SciPy 1.16.1 and openpyxl 3.1.5. Local validation uses Python
3.13.7. Use Python 3.11–3.13 on Colab. No unconstrained package upgrade is part
of the notebook. Transitive packages are resolved by pip and their actual
versions should be retained with the session's `pip freeze` output.

The notebook installs the exact `torch==2.12.0+cu126` CUDA wheel first, following the
[official PyTorch wheel recipe](https://pytorch.org/get-started/previous-versions/#v2120),
then installs the requirements. It removes unused torchvision/torchaudio/torchtext
extensions to avoid stale binaries after replacing torch. This changes only the
disposable Colab environment. The CPU-verified set plus this CUDA wheel is a
**candidate Colab environment**, not a claim of completed CUDA validation.
[Transformers installation guidance](https://huggingface.co/docs/transformers/v5.8.1/installation)
also requires a working PyTorch/CUDA runtime for GPU acceleration. Production
preflight must pass on the actual GPU before fit; retain errors if it does not.
Do not substitute an unrecorded upgrade or claim GPU compatibility from CPU tests.

The session-evidence cell records actual versions, a full `pip freeze` listing,
Python, code SHA, CUDA runtime,
GPU, mode and run ID before execution. Its unique JSON goes to `/content` and
the `sessions/` directory beside the durable `runs/` root, without reserving a
run directory. It deliberately records `production_training_completed: false`:
that setup snapshot never certifies later completion. `execution.json` records
the workflow's frozen actual dependency/GPU environment. Preserve the setup JSON
with the executed notebook for transitive dependency evidence.

| Setting | Required location/meaning |
| --- | --- |
| `source` | Original Kim 2018 XLSX or supported CSV/TSV, accessible from this session; no personal path is required |
| `partition_path` | Reusable manifest outside individual runs, preferably on Drive; existing membership is verified, never silently resplit |
| `split` | Production `SplitConfig()`: 12,000/1,500/1,500, seed 42; smaller sizes belong only to explicit synthetic fixtures |
| `model_source` | Original generic NT-500M reference or immutable SHA; preflight resolves and freezes the exact revision |
| `local_work_root` | VM `/content/cas12a-nt-work` or another `/content` directory outside the Drive mount |
| `drive_mount_root` | Authorized mount, default `/content/drive`; update it if mounting elsewhere |
| `drive_output_root` | Durable mounted Drive `runs/` root; disjoint from local work |
| `run_id` | Explicit new ID for fresh mode; local and durable existing destinations are refused |
| `resume_run_id` | Original run ID for resume mode; use the same source, partition, roots, code and runtime |

The returned `run.checkpoint_dir` is `<local_work_root>/<run_id>/checkpoints`.
Checkpoint locations are derived from the configured local root, separately
from Drive output and the reusable partition. Local work uses VM storage because
[Colab warns about Drive I/O and VM lifetime](https://research.google.com/colaboratory/faq.html).
Drive receives selected weights/tokenizer and small evidence, never routine
multi-GB optimizer/RNG checkpoints.

## Lifecycle and permitted continuation

The notebook calls these same public operations, in this order:

```python
run = workflow.prepare(run_id=run_id)  # Fresh mode only.
run = workflow.preflight(run, model=model_source)  # Fresh mode only.
run = workflow.fit(run)  # OR workflow.resume(resume_run_id), explicitly.
run = workflow.export_selected(run)
run = workflow.verify_export(run)
run = workflow.persist_export(run)
run = workflow.verify_durable_export(run)
run = workflow.evaluate(run)
run = workflow.persist_evaluation(run)
run = workflow.verify_durable_evaluation(run)
```

Preflight runs two optimizer updates and validation, including backward after
optimizer state allocation, and freezes resource-feasible settings. The fit
starts fresh rather than using preflight-trained weights. Full fine-tuning uses
effective batch 32, 125-update validation/save intervals, at most 10 epochs,
lowest-validation-MSE selection and patience three. Inspect the recorded recipe
instead of changing it in notebook cells. See [preflight](nt-preflight.md),
[training](nt-training.md), [durable export](nt-durable-export.md) and
[final evaluation](nt-evaluation.md) for the authoritative contracts.

| Observed condition | Next action |
| --- | --- |
| `prepared` | Run production preflight; this is data readiness only |
| `preflight_failed`, CUDA unavailable or `ResourceConstraint` | Preserve preflight attempts, actual environment and traceback; stop. No smaller backbone, CPU production fit or silent recipe change. An explicitly chosen future retry needs a new recorded run ID and the same partition |
| `preflight_passed` | Explicitly start fit; no production model exists yet |
| `training_interrupted`/`training_failed`, or abandoned `training`/`resuming` | Inspect same-VM checkpoints, restore the original settings, choose resume mode and original ID. Resume verifies complete latest state and preserves optimizer, RNG, selection and patience progress |
| `resume_failed` | Preserve error and evidence; repair only the identified external cause. Missing/damaged/incompatible state must not be bypassed or replaced with an older checkpoint silently |
| `recovery_unavailable`, VM reset or no complete local checkpoint | No recovery from Drive selected exports/status. Preserve the old durable run; explicitly choose fresh mode with a new ID, same source and existing verified partition. No automatic fit or resplit |
| `training_completed` | Continue local export/reload, then durable export, then final evaluation; do not call fit/resume again |
| Disk exhaustion | Preserve incomplete saves and best/latest checkpoints. Stop; inspect available VM space. Retention is at most best/latest completed states and does not eliminate temporary-save headroom requirements |
| Drive disconnect, copy failure, `.incomplete` destination | Stop; preserve local failure/status evidence and traceback, reconnect/authorize Drive and inspect both stores. Do not delete destinations to force a retry or treat an incomplete copy as valid. Publication retry/reconciliation is not automated; investigate before continuing |
| `export_failed`/`durable_export_failed` | Do not evaluate or claim durability. Inspect reload/file/receipt identities. Preserve the existing output; never retrain automatically to repair export |
| `final_evaluation_validated` | Local results only; publish and verify the durable evaluation |
| `final_evaluation_failed` | Preserve numerical/identity failure evidence. Raw collapse and nonfinite results cannot complete. Do not retune or refit based on final values |
| `final_evaluation_completed` | Reverify durable artifacts and collect evidence. No extra experiment is started |

Resume is for training interruptions only; a finally evaluated/exported run
cannot become a training input. [Recovery details](nt-resume.md) describe the
eligibility checks. Unsaved updates after the latest completed checkpoint may
be replayed. Re-executing completed cells is not a generic recovery mechanism.

## Required artifacts and actual Colab evidence

Inspect the local and durable copies. `status.json` is the current stage;
`lifecycle.json` preserves timestamped transitions and failures. A stale notebook
variable or console message is insufficient evidence after an exception.

| Artifact | Evidence to inspect |
| --- | --- |
| `data.json`, `partition.json`, `configuration.json` | Original source/table hashes, 15,000 usable rows, exact complete disjoint split, training-only frozen baseline and configured roots |
| `tokenization.json` | Original-revision tokenizer, all 34 input bases preserved, no truncation/unknowns and maximum 12 padded positions |
| `preflight.json`, `execution.json` | `production_gpu`, two completed probe updates, finite post-Adam backward and validation, actual GPU, package versions, immutable backbone/tokenizer SHA, precision, microbatch/accumulation/checkpointing, warmup, peak memory and timings |
| `training.json`, `history.json` | Frozen optimizer/schedule, real optimizer-update and validation histories, selected update/fractional epoch, actual update/epoch counts and stopping reason |
| Local `checkpoints/` | Complete best/latest manifests and training state, at most two completed checkpoints; absent from durable output |
| `export.json`, `selected_export/` | Selected checkpoint/artifact hashes, matching saved model/tokenizer, finite fresh-instance reload predictions with recorded atol/rtol and consistent tokenization |
| `durable_export.json` | Validated copied file inventory and matching selected identity; recheck with `verify_durable_export` |
| `final_evaluation/{evaluation.json,metrics.json,predictions.csv}` | Validated frozen selected artifact on exactly 1,500 targets; per-target source/sequence identity, labels, raw, bounded and baseline predictions; raw/clipped/baseline metric namespaces, diagnostics and undefined-correlation reasons |
| `durable_evaluation.json` | Verified final bundle hashes tied to the same selected checkpoint/artifact/partition; recheck with `verify_durable_evaluation` |
| `status.json`, `lifecycle.json` | Production completion only after local reload, durable export and final evaluation publication all pass |

After the notebook's public verification calls, independently check the copied
files from the mounted Drive using the receipts. This read-only evidence step
belongs in a terminal or separate audit session; it does not implement training
or change workflow state:

```python
import hashlib
import json
from pathlib import Path

durable = Path("/content/drive/MyDrive/cas12a-nt/runs/YOUR_RUN_ID")
for receipt_name, root in (
    ("durable_export.json", durable),
    ("durable_evaluation.json", durable / "final_evaluation"),
):
    receipt = json.loads((durable / receipt_name).read_bytes())
    assert receipt["validated"] is True
    for relative, expected in receipt["files"].items():
        with (root / relative).open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == expected
```

Keep the executed notebook (including errors/skipped stages), setup JSON and
freeze listing, code SHA, run ID, durable location, verified receipts and stage
artifacts. Record human session start/Drive authorization, actual GPU, execution
dates and the exact last completed stage. Report unavailable GPU/Drive execution
as **unexecuted**. Do not manually edit status flags to represent a desired result.

The production completion checks are `run_kind: production`,
`state: final_evaluation_completed`, `local_export_validated: true`,
`durable_export_validated: true`, `durable_evaluation_validated: true`,
`final_evaluation_completed: true` and `model_completed: true`, backed by the
artifacts above and successful reverification. Valid low metrics do not fail a
superiority gate. CNN five-fold Spearman around 0.751 is historical context:
its population/procedure differs and existing CNN checkpoint training overlaps
this holdout, so it is not a leakage-free direct comparison.

## Ordinary offline verification

With the existing scientific Python environment, run:

```text
python -m pytest cas12a/test/test_nt_colab.py -q
python -m pytest cas12a/test -q
```

The notebook integration test executes the actual workflow cells using the same
public interface, synthetic source, temporary local/durable destinations and a
tiny locally saved ESM model/tokenizer. It needs no pretrained download, Drive,
Colab session or GPU. It checks training, selected reload, final artifacts and
copied hashes. Its status stays `run_kind: fixture`, `model_completed: false`.
These are not production training metrics. The existing recovery tests cover
interruption and compatible/missing/corrupt local state at the same public seam.
The subtree run includes the existing CNN regressions; a full-repository suite
is unnecessary for this handoff.
