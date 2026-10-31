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
checkpoint manifest hashes. Small configuration/history/status evidence is
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
Compatible local resume is a later lifecycle operation; another `fit` does not
silently restart an interrupted or completed run.

`export_selected` verifies the selected checkpoint hashes, loads its model and
matching original-revision tokenizer, and checks validation predictions against
the selected training history using the exact sample and batch shape recorded
at selection, avoiding mixed-precision rounding changes from different evaluation
batch shapes. It writes weights/config/tokenizer to the fresh
local `selected_export/` directory, frees that model instance and reloads both
components from the export. It requires identical validation-sample token IDs,
masks and vocabulary, finite predictions, and `abs(a-b) <= 1e-6 + 1e-5 * abs(b)`.
It checks up to eight validation targets and records their identities, both
prediction sets, numerical tolerance, maximum absolute difference, file hashes
and aggregate selected-artifact hash in `export.json`. This tolerance permits
ordinary numerical differences; it does not require cross-device bit identity.
Reverification checks the saved evidence/file identities and fresh reload again.
Missing, damaged, inconsistent or failed reloads record `export_failed` and cannot
claim a validated local export. Existing export destinations are never overwritten.

Training completion and `export_validated` describe separate local stages.
`durable_export_validated` and `model_completed` remain false at this point.
Continue with the [durable export workflow](nt-durable-export.md) to copy and
verify the selected export on Drive. Numerical final Testset evaluation belongs
to [#107](https://github.com/shin584/project/issues/107). Neither fixture checks nor
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
Run `python -m pytest cas12a/test/test_nt_training.py -q`, then
`python -m pytest cas12a/test -q` for the affected subtree.
