# NT-Cas12a training and local selected export

Issue [#105](https://github.com/shin584/project/issues/105) extends the
[preflight lifecycle](nt-preflight.md) through bounded fitting, validation
selection and local reload verification. Use the same prepared workflow/run:

```python
run = workflow.preflight(run)  # CUDA required for production
run = workflow.fit(run)
assert run.status["state"] == "training_completed"
run = workflow.export_selected(run)
assert run.status["state"] == "export_validated"
run = workflow.verify_export(run)  # Recheck before consuming a local export.
```

`fit` recreates the generic initialization, seeded RNG, empty AdamW state,
unstepped scheduler and fresh precision scaler using the frozen preflight
identity checks. It never accepts an already stepped model. Initialization and
data ordering use seed 42. AdamW uses learning rate 2e-5, weight decay 0.01,
betas (0.9, 0.999), epsilon 1e-8 and `foreach=False`. The single-GPU effective
batch is 32, using the frozen microbatch/accumulation decomposition and precision.
All encoder and head parameters train; gradients are clipped at norm 1.0.
Microbatch MSE contributions are weighted by their number of examples, including
the last partial fixture batch. An FP16 overflow backs off the scale and retries
the same effective batch, at most 32 retries, without advancing optimizer or
scheduler progress. Persistent nonfinite results fail explicitly.

The linear schedule is fixed before fitting, with at most 10 epochs and
`ceil(0.05 * planned_optimizer_updates)` warmup updates. Production has 375
updates per epoch, 3,750 planned updates and 188 warmup updates. Early stopping
never recalculates these values. Data ordering uses successive epoch permutations
from `numpy.random.default_rng(42)` over the saved ordered training membership.

Validation and checkpoint creation occur every 125 **successful optimizer
updates after accumulation**. The strictly lowest validation MSE selects the
checkpoint; a tie counts as no improvement. Three consecutive non-improving
validations stop fitting. Training and validation histories record global steps,
fractional epochs, measured losses, learning rates and overflow retries, the
selected point, actual updates/epochs and stopping reason. Training, selection
and export verification use only training/validation targets. Testset labels
are never used by these operations.

`training.json` records resolved optimizer, scheduler, scaler, device, batch,
precision, clipping, selection, seed and trainer settings with the frozen
execution identity. `history.json` links to that recipe and the selected/latest
checkpoint manifest hashes, and records the selected point's reload sample
predictions and `selected_weight_identities` taken from the live model. Small configuration/history/status evidence is
verified in both local and durable destinations at stage boundaries.

Local `checkpoints/checkpoint-<global_step>/` directories contain:

- Model weights and config, saved with safetensors.
- `training_state.pt`: AdamW moments, scheduler, applicable FP16 scaler,
  Python/NumPy/PyTorch/CUDA RNG and data-order generator states, epoch permutation
  and next example offset, optimizer-update position, selection/patience progress
  and history.
- `checkpoint.json`: complete marker, file hashes, recipe/configuration/execution
  identities, global step and measured disk headroom.

Creation checks free space for model and Adam state plus 10% and 64 MiB overhead
before writing. Files first go into a distinct `.incomplete` directory; only a
completed save receives its final checkpoint name. Retention then preserves the
best-validation and latest completed checkpoints, at most two. Interrupted writes
remain identifiable as incomplete. Selected export and evidence are outside this
pruning area. Routine full checkpoints never go to the durable/Drive destination.
Use the explicit [same-VM resume operation](nt-resume.md) for compatible local
recovery; another `fit` does not silently restart an interrupted or completed run.

`export_selected` verifies the selected checkpoint hashes, loads its model and
matching original-revision tokenizer, and checks validation predictions against
the selected training history using the exact sample and batch shape recorded
at selection, avoiding mixed-precision rounding changes from different evaluation
batch shapes. It writes weights/config/tokenizer to the fresh
local `selected_export/` directory, frees that model instance and reloads both
components from the export. It requires identical validation-sample token IDs,
masks and vocabulary and finite predictions. It checks up to eight validation
targets and records in `export.json` their identities, both prediction sets, the
per-tensor weight digests, the applied `forward_tolerance` and its `precision`,
each comparison's `max_absolute_difference`/`max_relative_difference`, the file
hashes and the aggregate selected-artifact hash.
Reverification checks the saved evidence/file identities and fresh reload again.
Missing, damaged, inconsistent or failed reloads record `export_failed` and cannot
claim a validated local export. Existing export destinations are never overwritten.

### Weight identity and forward reproducibility are judged separately

Issue [#111](https://github.com/shin584/project/issues/111) splits the single
reload comparison into two assertions of different kinds, because one tolerance
cannot serve both.

Weight identity is **exact**, and it is anchored on the **live training model at
selection**. When a validation step improves, `fit` records every parameter and
buffer of the in-memory model as shape, dtype and the SHA-256 of its raw bytes
into `history.json` as `selected_weight_identities`. Both later loads — the
selected checkpoint in `export_selected`, and the fresh instance reloaded from
`selected_export/` — are compared against that recording, before any device copy
or forward pass. Raw bytes are hashed through a `uint8` view, so the check needs
no NumPy equivalent for the dtype and never casts or rounds a value.

The anchor has to be the live model rather than one load compared with another:
`from_pretrained` quietly re-initializes keys a checkpoint is missing, so a
checkpoint load that dropped its classification head would agree perfectly with
every later reload of itself. Against the recorded selection state it cannot.
A re-initialized head, an altered encoder weight, a dropped tensor and an extra
tensor all fail here no matter how the forward pass lands, and the failure names
the differing tensors. `export.json` carries the same per-tensor digests with
their aggregate hash, and reverification refuses a report that substitutes its
own anchor for the recorded one.

Forward reproducibility is **approximate**, and its bound is derived from
`execution.json`'s frozen `settings["precision"]` rather than from a global
constant. Two instances holding identical weights are not guaranteed the same
kernel or reduction order, and the resulting drift scales with the autocast
compute dtype's unit roundoff — half the gap between 1.0 and the next
representable value. The derivation is `rtol = 128 * unit_roundoff`, with
`atol = rtol / 10` to anchor logits near zero.

The 128-step budget is a **deliberate margin, not a measured error bound**: no
real FP16 kernel comparison has been run, so the figure is chosen to avoid
false failures rather than fitted to observed drift. That is defensible only
because the exact weight check above carries loading correctness on its own —
the forward bound never has to detect a bad load, so erring wide costs nothing
that matters. Were the forward comparison still the sole correctness signal, a
6% (FP16) or 25% (BF16) accepted logit shift would be far too loose to mean
anything.

| `precision` | Unit roundoff | `rtol` | `atol` |
| --- | --- | --- | --- |
| `float32` | 2⁻²⁴ ≈ 5.96e-8 | 7.63e-6 | 7.63e-7 |
| `float16` | 2⁻¹¹ ≈ 4.88e-4 | 6.25e-2 | 6.25e-3 |
| `bfloat16` | 2⁻⁹ ≈ 1.95e-3 | 2.50e-1 | 2.50e-2 |

The previous fixed `1e-6`/`1e-5` pair was structurally unsatisfiable under FP16:
a single representable step near 1.0 is about 9.8e-4, three orders of magnitude
wider than that absolute bound. T4-class GPUs have no native BF16, so FP16 is
forced, and a completed production fit was therefore blocked at export. The
adopted criterion is recorded in `export.json` as `forward_tolerance` with its
basis, and reverification refuses evidence whose recorded tolerance does not
match the one its precision implies.

### Failed reload comparisons keep their numbers

A failed comparison raises `ReloadMismatch`, and the workflow appends its
evidence to `export_failure.json` in both the local and durable stores before
committing `export_failed`. Each entry holds the failure kind
(`forward_reproducibility` or `weight_identity`), which comparison failed, both
prediction arrays, the maximum absolute and relative deviation, and the applied
tolerance and precision — or, for a weight mismatch, the differing tensors with
their shapes, dtypes and digests. A non-finite value is written as its `repr`
(`'nan'`, `'inf'`) rather than dropped, because canonical evidence forbids JSON
NaN literals and a divergence is exactly where the magnitude matters. Earlier
entries are never replaced, so a retried export cannot erase why the previous
attempt failed. This evidence is what makes a cause determinable after the VM
holding the checkpoint is reclaimed.

### Retrying a failed export

```python
run = workflow.retry_export(run)  # Only from export_failed.
```

`retry_export` re-exports the same completed training. It re-verifies the
checkpoint, execution and training identities from scratch, records
`export_retry: true`, and never refits: a failed export is a reason to
re-export, never to retrain. It refuses to run while a partially created
`selected_export/` directory or an `export.json` from an earlier attempt still
exists in either store — reclaiming those destinations is a deliberate operator
decision, not something the retry performs on their behalf. `lifecycle.json`
keeps the failed attempt alongside the retry.

The `export_failed` production run reported in
[#110](https://github.com/shin584/project/issues/110) predates this evidence,
and its checkpoints were lost with the VM. Whether its original failure was a
numerical variation or a real loading defect **cannot be determined**, and it is
not claimed to have been harmless. Its preserved evidence stands as recorded and
is not reinterpreted here; the re-execution runs under a new `run_id` with the
existing Drive `partition.json`.

Training completion and `export_validated` describe separate local stages.
`durable_export_validated` and `model_completed` remain false at this point.
Continue with the [durable export workflow](nt-durable-export.md) to copy and
verify the selected export on Drive, then run the
[final Testset evaluator](nt-evaluation.md). Neither fixture checks nor
this implementation establish real NT-500M GPU training or a measured model
outcome. Colab execution remains unexecuted until actual GPU evidence is saved.

Offline runs use the same public boundary with a tiny locally saved ESM model,
six-mer tokenizer and synthetic source. Shorter validation intervals and optional
controlled selection-loss histories must be explicitly labeled:

```python
from cas12a.nt_workflow import FitFixture, ModelSource

run = fixture_workflow.preflight(run, model=ModelSource.fixture(tiny_model_path))
run = fixture_workflow.fit(
    run, fixture=FitFixture(interval_updates=1, validation_losses=(0.2, 0.4, 0.3, 0.2))
)
run = fixture_workflow.export_selected(run)
```

Production rejects fixture controls. The optional losses affect only selection
in labeled fixtures; real validation predictions, finiteness and measured MSE
are still computed and recorded. No experiment performance threshold is asserted.

The FP16 regression cases **relabel** the frozen precision on a CPU fixture that
still computes in float32, so they verify which criterion is applied and that an
FP16-magnitude forward shift passes it while the float32 bound rejects the same
shift. They do not exercise real FP16 kernels; no offline test can, and the
128-step budget therefore rests on no measurement.
Run `python -m pytest cas12a/test/test_nt_training.py -q`, then
`python -m pytest cas12a/test -q` for the affected subtree.
