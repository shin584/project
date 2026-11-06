# #143 implementation and verification

Date: 2026-11-06. Branch: `model_analysis`. Starting point:
`f2c51e28239047afd630266a4e850ddf85729782`.

The user requested `implement #143` and confirmed the computation, artifact
save/read and durable-driver test boundaries and explicit policy with `y`.
The [CPU contract](../planning/issue143-rollout-contract.md) passed independent
Standards and Spec design review with no blockers before implementation.
Diagnostic thresholds in synthetic fixtures do not approve real NT thresholds.
Scientific freeze/actual validation remain #146 and actual 1,500-row Testset
rollout remains #147.

## Implementation

`nt_rollout.compute_rollout` uses the checked native scalar ESM regressor in
eval FP32 eager attention with autocast/TF32 disabled. It verifies fresh
native/saved and attention-enabled/native predictions at atol=1e-5, rtol=1e-4.
Every actual returned layer must contain finite, nonnegative FP32 probabilities
of the configured head count and `(1, heads, 12, 12)` shape, normalized rows
within the parity tolerances, and no padding-key mass within atol=1e-5. Backend
class, package versions, layer/head counts and actual layer order are recorded.

The head mean excludes padding coordinates before propagation. Residual mixing
is `0.5*A+0.5*I`, followed by row normalization and left multiplication in
encoder order. CLS stays in every float64 product. Per-source NPZ artifacts
retain all valid-token attention means `(layers,10,10)`, full rollout `(10,10)`
and the unnormalized sequence block `(9,9)`. DNA spans, original token positions,
input IDs, attention masks and special masks remain available in JSON.

Removed CLS mass, retained sequence row mass and full row sums are reconstructed.
The display block is never implicitly renormalized. All 36 unordered DNA-query
pairs are compared by L1 over full valid-key support and separately over the
display block. The explicit `query_l1_threshold` policy determines the stored
minimum/mean/maximum, near-identical fraction and status. Near-identical queries
remain valid results with an attention-flow interpretation limitation; they
do not establish output attribution or selective token-pair interactions.

`nt_rollout_artifacts` commits an immutable all-prediction-ID plan before model
loading. Shards bind prediction/source/token identities, model/tokenizer/code/
environment/protocol hashes, backend metadata, producing verified session,
axes, quantity, normalization and quality. Layer/head counts are independently
checked against hash-verified config snapshots and package versions against
the producing environment. Readers verify committed bytes and reconstruct
parity, propagation, full/block mass and diagnostics without torch,
Transformers, original source files or model access. Uncommitted files are
ignored. No regression-output-head axis or base-coordinate expansion is added.

Numerical failures retain finite observed outputs, null arrays and a stage and
reason. Diagnostic coverage is separate from accepted coverage; failed sources
block completion and automatic retries. Budget, interruption, model-state and
storage faults propagate through existing unresolved-job or poisoned-manager
semantics. `analyze.run_rollout` shares the #139 prediction run, cumulative
ledger, session parity, explicit termination settlement and sequential-writer
assumption. It commits whole-source shards after state checks and reuses only
verified committed coverage. Protocol and contract are included in execution
identity snapshots; earlier code identities require replacement runs retaining
the same study charges.

## Verification

Focused red/green checks covered calculation, storage and driver boundaries.
A synthetic noncommuting two-layer attention model checks head averaging,
product direction, padding exclusion and exact CLS mass. A saved tiny scalar
ESM verifies actual attention-returning eager inference and FP32 parity.
Attention absence, layer/head shape defects, nonfinite/negative probabilities,
padding-key mass, changed native prediction and attention-enabled parity
failure are distinguished from budget/execution exceptions. Explicit policy
validation and near-identical-query classification are checked independently.

Storage checks cover source ordering, quality/provenance tampering, config
dimensions, package versions, corrupted/missing arrays, failure after NPZ write,
saved failed diagnostics and subprocess reload with model imports blocked and
original assets unavailable. Driver checks cover immutable policy, plan commit
before loading, explicit interrupted-session settlement, unchanged first-shard
hash and values, reuse without loading, cumulative 900-second fixture charges,
corruption before loading and quality failure blocking automatic retries.

Focused results: computation 11 passed before the final policy checks were
added; artifact 15 passed; driver 2 passed. An earlier driver run was invalidated
by source formatting during execution, correctly triggering code-identity
rejection; it passed after code was held stable.

Final analysis-subtree command from the repository root:

```text
python -m pytest xai_test/cas12a_NT_analysis/test -q
154 passed in 275.17s (0:04:35)
```

Ruff check/format and staged whitespace verification passed for the changed
files. The full repository suite was not run.

## Implementation review

The `code-review` skill reviewed only this task's staged diff against the
starting commit; unrelated working-tree changes were excluded. Both axes ran
as independent read-only agents.

### Standards

0 actionable findings. No documented-standard breaches or actionable baseline
smells. Config/version bindings, narrow numerical failure handling, checked
readback and agreed test boundaries satisfy the documented contract.

### Spec

0 actionable findings. The implementation covers the confirmed CPU scope of
#143 without changing the scientific-freeze or actual execution scope.

Total findings: Standards 0; Spec 0. Neither axis has a remaining issue.

## Execution limits

No real NT GPU execution, new FP32 Test prediction or retraining was performed.
No existing dataset split, checkpoint, evaluation artifact, historical ADR or
GitHub issue was modified. Only the touched analysis subtree is tested; the
full repository suite is excluded. Current execution entry points remain
validation-only. Existing sequential-writer, external termination-evidence
trust and storage acknowledgement limits from #139 still apply.
