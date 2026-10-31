# NT-Cas12a durable export and lifecycle evidence

Issue [#106](https://github.com/shin584/project/issues/106) adds durable storage
to the [local selected-model lifecycle](nt-training.md). On Colab, mount Drive
first, then configure `local_work_root` beneath `/content` on the VM (outside
the Drive mount) and `drive_output_root` beneath the mounted Drive. Both roots
are explicit, non-overlapping locations. Every run reserves a fresh directory
under each root; an existing destination is an error, including an incomplete
copy. Routine resumable training checkpoints stay in the VM's run directory.
`drive_mount_root` defaults to `/content/drive`; set it explicitly for a custom
mount. The workflow refuses to place local work anywhere beneath that mount,
even when the two run roots would otherwise be non-overlapping.

```python
run = workflow.prepare()  # fresh run_id; never reuse an existing destination
run = workflow.preflight(run)
run = workflow.fit(run)
run = workflow.export_selected(run)  # local fresh-instance model/tokenizer reload
run = workflow.persist_export(run)  # verified selected export to Drive
run = workflow.verify_durable_export(run)
assert run.status["durable_export_validated"]
assert not run.status["final_evaluation_completed"]
```

`persist_export` rechecks the local model/tokenizer reload and identity before
copying. It copies only `selected_export/`, containing selected weights, model
config and matching tokenizer. The data/partition manifests, configuration,
tokenization, frozen execution configuration (including package/GPU metadata),
preflight, resolved training recipe, training history and local reload report
are already synchronized to Drive. It verifies their hashes too.

Large copies first enter `.selected_export.incomplete/`. The workflow compares
the complete source and destination file/hash inventories, then publishes the
final directory and rechecks it. Only then does it write `durable_export.json`
and the `durable_export_validated` status. The receipt records per-file SHA-256,
the aggregate selected-artifact identity, selected checkpoint identity, local
reload report identity, run kind and verification timestamp. Optimizer/RNG
checkpoints are excluded. `verify_durable_export` rechecks the saved copies and
receipt without loading the training runtime; damaged or missing evidence
revokes durable validation.

`status.json` is the current stage marker. `lifecycle.json` preserves timestamped
status snapshots at stage boundaries: preparing/prepared, preflighting/passed or
failed, training/interrupted/failed/completed, local exporting/validated/failed,
and durable persisting/validated/failed. Small evidence uses verified temporary
files before replacement. Failure recording tries both stores independently;
storage errors are attached as notes to the original exception. A disconnected
Drive can leave only local failure evidence. Neither local nor durable status is
a guarantee that an unavailable storage device accepted a write.

Validated export is distinct from final Testset evaluation. The
[final evaluator](nt-evaluation.md) writes a local `final_evaluation/` bundle via
`workflow.evaluate(run)`. Call `workflow.persist_evaluation(run)` after its
scientific checks pass. The bundle contract is:

- `metrics.json`: the evaluator's final model and baseline metrics.
- `predictions.csv`: its per-target predictions and identities.
- `evaluation.json`: `validated: true`, `run_id`, `run_kind`, the matching
  `selected_checkpoint_sha256`, `selected_artifact_sha256`,
  `partition_manifest_sha256`, and a `files` map containing the SHA-256 hashes
  of `metrics.json` and `predictions.csv`.

Publication checks those identities, the input hashes, the durable selected
export, and the copied bundle using the same copy procedure. It records
`durable_evaluation.json` and marks `final_evaluation_completed` only after the
copy is verified. `verify_durable_evaluation` checks both the selected export
and evaluation bundle and revokes evaluation completion on corruption. This is
the storage handoff; this issue does not implement numerical Testset evaluation
or decide whether a model has collapsed. The evaluator owns those checks before
setting its report's `validated` field. Fixture publication remains labeled
`fixture` and never sets `model_completed` to true.

Run `python -m pytest cas12a/test/test_nt_durable.py -q` for offline integration
checks with temporary local/durable directories and a real tiny saved model.
These require no Drive connection or pretrained downloads. Then run
`python -m pytest cas12a/test -q` for the affected subtree.

Actual Colab/Drive validation is **unexecuted** in this implementation session.
It requires a human to start the GPU session and authorize the Drive mount.
Record the real run's preflight, execution/GPU metadata, history, local reload,
durable receipts and lifecycle evidence before claiming successful GPU training
or Drive export. After publication, independently verify the copied hashes from
the mounted Drive and preserve the run ID and destination as execution evidence.
Offline fixtures establish implementation behavior only. Durable selected
exports do not contain the complete training state needed for resume after VM
loss.
