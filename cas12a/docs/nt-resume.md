# NT-Cas12a same-VM recovery

Issue [#108](https://github.com/shin584/project/issues/108) adds explicit recovery
through the same `NTWorkflow` used to prepare and fit the run:

```python
# Recreate workflow with the original source, split and output locations if
# the Python handle was lost. The same Colab VM must still hold checkpoints.
run = workflow.resume("original-run-id")  # Or workflow.resume(run).
run = workflow.export_selected(run)
run = workflow.persist_export(run)
```

`resume` accepts an interrupted/failed training run, or one still marked
`training`/`resuming` after a process interruption. It verifies source-file and
canonical-table digests, the existing partition manifest and identical split
membership, immutable backbone/tokenizer revision, tokenization, frozen execution
settings, dependency/GPU environment and the resolved training recipe. It does
not accept new fit controls. The original fixture controls, where applicable,
come from the recorded recipe.

Only the latest completed `checkpoints/checkpoint-<update>/` is eligible. A
damaged latest checkpoint is rejected explicitly rather than silently choosing
an earlier checkpoint. Hidden `.incomplete` saves are ineligible. Recovery checks
the checkpoint manifest and file hashes, model/config, complete AdamW moments,
scheduler position, Python/NumPy/PyTorch/CUDA and data-order RNG states, applicable
FP16 scaler, epoch permutation and next-example offset, validation selection and
early-stopping progress. The selected best-validation checkpoint must also remain
intact. Missing, corrupt or incompatible state records `resume_failed` and raises.
Missing local training state raises the public `RecoveryUnavailable` exception
(a `FileNotFoundError`), without importing the model runtime just to report loss.

The continuation restores successful optimizer-update position, accumulation
batch boundaries, scheduler, random state, training/validation history and
non-improvement count. A completed early-stopping checkpoint does not take extra
updates. The selected checkpoint can remain earlier than the resumed checkpoint.
Unsaved work after the latest completed checkpoint is replayed. An interrupted
save remains identifiable, and a retry stages a separate incomplete directory.
Successful saves retain at most the best and latest completed checkpoints;
selected exports and durable evidence remain outside the pruning area.

Small durable `status.json` and `lifecycle.json` records retain interruption,
failure, `resuming` with checkpoint identity/update position, and completion.
No routine full-checkpoint Drive backup is added. A selected export or durable
status alone is never a resumable training state. Recovery cannot continue an
exported or finally evaluated run.

When local state is absent, `resume` raises with `recovery unavailable` and
records that state durably where possible, preserving the existing lifecycle
history even if the local directory disappeared. There is no guarantee of
recovery after Colab VM loss. Starting over requires an explicit new run:

```python
fresh = workflow.prepare(run_id="explicit-new-run-id")
# prepare verifies and reuses the same partition; it reserves new run directories.
fresh = workflow.preflight(fresh)
fresh = workflow.fit(fresh)
```

No automatic fresh fit, resplit, or final-result-driven retuning occurs. These
offline checks establish workflow readiness, not real NT-500M Colab training.
Run `python -m pytest cas12a/test/test_nt_resume.py -q`, then
`python -m pytest cas12a/test -q` for the affected subtree.
FP16 scaler continuation and missing/partial-state cases run only on a CUDA
runtime whose resolved precision is FP16; CPU and BF16 runtimes report those
checks skipped rather than claiming FP16 execution.
